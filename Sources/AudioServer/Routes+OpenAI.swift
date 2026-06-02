import Foundation
import Hummingbird
import NIOCore
import VoxCPM2TTS

// MARK: - OpenAI-Compatible REST Routes
//
// `POST /v1/audio/speech` in the shape OpenAI's HTTP TTS API documents:
//   https://platform.openai.com/docs/api-reference/audio/createSpeech
//
// Clients that target OpenAI's `/v1/audio/speech` (voicemode, openai-python,
// many integration libs) can point at this server unchanged.
//
// Single engine: VoxCPM2 (2B, 48kHz). Voice selection happens via the `voice`
// field; see the routing rules in `handleOpenAISpeech` for the precedence.

/// Per-model variant cache. Lets clients select a VoxCPM2 size/precision via
/// the OpenAI `model` field. Each variant is a couple of GB of MLX weights,
/// so we keep loaded variants resident across requests.
private let modelCache = VoxCPM2Cache()

actor VoxCPM2Cache {
    private var entries: [String: VoxCPM2TTSModel] = [:]

    func load(modelId: String?) async throws -> VoxCPM2TTSModel {
        let key = modelId ?? "default"
        if let m = entries[key] { return m }
        let m: VoxCPM2TTSModel
        if let modelId = modelId {
            m = try await VoxCPM2TTSModel.fromPretrained(modelId: modelId)
        } else {
            m = try await VoxCPM2TTSModel.fromPretrained()
        }
        entries[key] = m
        return m
    }
}

func registerOpenAIRoutes(on router: Router<BasicRequestContext>, state: ModelState) {
    router.post("/v1/audio/speech", use: handleOpenAISpeech(state: state))
}

private func handleOpenAISpeech(
    state: ModelState
) -> @Sendable (Request, BasicRequestContext) async throws -> Response {
    return { request, _ in
        let body = try await request.body.collect(upTo: 1024 * 1024)
        guard let json = try JSONSerialization.jsonObject(with: Data(buffer: body)) as? [String: Any] else {
            return errorResponse("Body must be JSON", status: .badRequest)
        }
        guard let input = (json["input"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !input.isEmpty else {
            return errorResponse("'input' must be a non-empty string", status: .badRequest)
        }
        guard input.count <= 4096 else {
            return errorResponse("'input' must be 4096 characters or fewer", status: .badRequest)
        }

        let voiceRaw = (json["voice"] as? String) ?? ""
        let responseFormat = ((json["response_format"] as? String) ?? "wav").lowercased()
        guard responseFormat == "wav" || responseFormat == "pcm" else {
            // OpenAI also documents mp3/opus/aac/flac. Synthesizing them would
            // need ffmpeg conversion of the raw float samples — out of scope.
            return errorResponse(
                "'response_format' must be 'wav' or 'pcm' (mp3/opus/aac/flac not supported yet)",
                status: .badRequest)
        }

        // `model` is the OpenAI hook for selecting model size/precision. We
        // treat it as an HF model ID when it matches a recognized VoxCPM2
        // prefix; anything else (incl. OpenAI's "tts-1"/"tts-1-hd") falls
        // through to the engine default.
        let modelId = (json["model"] as? String).flatMap(resolveVariantModelId)

        // OpenAI's gpt-4o-mini-tts API uses `instructions` for natural-language
        // style control ("speak excitedly", "in a low whisper"). VoxCPM2 takes
        // the same string via its `instruct` arg.
        let instructions: String? = (json["instructions"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }

        // Voice cloning (custom extension to OpenAI's shape). `clone_ref` is
        // either an absolute filesystem path or an https:// URL to a WAV file.
        // `clone_ref_text` (recommended) is the transcript of the reference
        // clip; with both VoxCPM2 uses its high-fidelity in-context prompt
        // path. Without `clone_ref_text` we fall back to audio-only ref
        // conditioning (lower fidelity).
        let cloneRef: String? = (json["clone_ref"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        let cloneRefText: String? = (json["clone_ref_text"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }

        // Routing precedence:
        //   1. Explicit `clone_ref` wins.
        //   2. Registry lookup on `voice`:
        //      - bare id ("ed_irish_m") → exact match
        //      - "claude_<hex>" / "blend_<hex>" → SHA-256 hashed across the
        //        sorted registry. Same session id → same voice across restarts.
        //   3. `voice=voxcpm2` (or `voxcpm2:...`) → bare VoxCPM2 default speaker.
        //   4. Anything else → bare VoxCPM2 default speaker.
        if let cloneRef {
            return try await handleVoxCPM2Clone(
                input: input,
                cloneRef: cloneRef,
                cloneRefText: cloneRefText,
                responseFormat: responseFormat,
                instructions: instructions,
                modelId: modelId)
        }

        let voiceLower = voiceRaw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let registryEntry: VoiceEntry? = {
            guard !voiceRegistry.isEmpty, !voiceLower.isEmpty else { return nil }
            if voiceLower.hasPrefix("claude_") || voiceLower.hasPrefix("blend_") {
                return voiceRegistry.hashedLookup(sessionId: voiceLower)
            }
            return voiceRegistry.lookup(id: voiceLower)
        }()
        if let entry = registryEntry {
            return try await handleVoxCPM2Clone(
                input: input,
                cloneRef: entry.refPath,
                cloneRefText: entry.refText,
                responseFormat: responseFormat,
                instructions: instructions,
                modelId: modelId)
        }

        return try await handleVoxCPM2Bare(
            input: input,
            responseFormat: responseFormat,
            instructions: instructions,
            modelId: modelId)
    }
}

// MARK: - Voice Cloning

/// Cache of decoded reference audio keyed by source path/URL. VoxCPM2 still
/// re-encodes the VAE features per call; only the audio decode + (when remote)
/// the HTTP fetch are memoized here.
private let refAudioCache = RefAudioCache()

actor RefAudioCache {
    private var entries: [String: [Float]] = [:]

    func load(ref: String) async throws -> [Float] {
        if let cached = entries[ref] { return cached }
        let data: Data
        if ref.hasPrefix("http://") || ref.hasPrefix("https://") {
            guard let url = URL(string: ref) else {
                throw CloneError.invalidRef("Invalid URL: \(ref)")
            }
            let (fetched, response) = try await URLSession.shared.data(from: url)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
                throw CloneError.invalidRef("Fetched \(ref): HTTP \(httpResponse.statusCode)")
            }
            data = fetched
        } else {
            // Relative paths would resolve against the server's cwd, which is
            // rarely what callers expect. Force absolute.
            guard ref.hasPrefix("/") else {
                throw CloneError.invalidRef(
                    "clone_ref must be an absolute path or http(s):// URL (got '\(ref)')")
            }
            data = try Data(contentsOf: URL(fileURLWithPath: ref))
        }
        // VoxCPM2's encoder runs at 16kHz.
        let samples = try decodeWAVData(data, targetSampleRate: 16000)
        entries[ref] = samples
        return samples
    }
}

enum CloneError: Error, LocalizedError {
    case invalidRef(String)

    var errorDescription: String? {
        switch self {
        case .invalidRef(let msg): return msg
        }
    }
}

/// VoxCPM2 voice clone. Given the reference clip *and* its transcript VoxCPM2
/// uses an in-context "prompt" (continue-this-voice) path which is materially
/// better than audio-only conditioning. Without the transcript we fall back
/// to the audio-only `refAudio` path.
///
/// Long inputs are split into sentences and synthesized per-sentence. VoxCPM2
/// destabilizes (speed-up, buzzing) on long single calls per the upstream
/// usage guide; per-sentence batching also cuts TTFB on multi-sentence inputs
/// because each sentence's PCM streams out as soon as it's ready.
///
/// Note: this is *not* the upstream Python `generate_streaming` API — the
/// Swift VoxCPM2TTS wrapper doesn't expose intra-sentence streaming. Inside a
/// sentence we still wait for the full buffer before writing. Adding true
/// intra-sentence streaming requires opening up the patch-decode loop in
/// `VoxCPM2TTSModel`.
private func handleVoxCPM2Clone(
    input: String,
    cloneRef: String,
    cloneRefText: String?,
    responseFormat: String,
    instructions: String?,
    modelId: String?
) async throws -> Response {
    let voxModelId: String? = (modelId?.contains("VoxCPM2") == true) ? modelId : nil
    let model = try await modelCache.load(modelId: voxModelId)
    let refSamples = try await refAudioCache.load(ref: cloneRef)
    let sampleRate = model.sampleRate
    let sentences = splitIntoSentences(input)
    let contentType = (responseFormat == "wav") ? "audio/wav" : "audio/pcm"
    let format = responseFormat

    return Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init { writer in
            do {
                if format == "wav" {
                    try await writer.write(
                        ByteBuffer(bytes: streamingWAVHeader(sampleRate: sampleRate)))
                }
                for sentence in sentences {
                    let samples: [Float]
                    if let cloneRefText {
                        samples = try await model.generateVoxCPM2(
                            text: sentence,
                            language: "english",
                            promptText: cloneRefText,
                            promptAudio: refSamples,
                            instruct: instructions)
                    } else {
                        samples = try await model.generateVoxCPM2(
                            text: sentence,
                            language: "english",
                            refAudio: refSamples,
                            instruct: instructions)
                    }
                    guard !samples.isEmpty else { continue }
                    try await writer.write(ByteBuffer(bytes: float32ToPCM16LE(samples)))
                }
                try await writer.finish(nil)
            } catch {
                try? await writer.finish(nil)
                throw error
            }
        })
}

/// Bare VoxCPM2 (no reference, no registry voice). Uses the model's default
/// speaker. `instructions` still passes through for style steering.
private func handleVoxCPM2Bare(
    input: String,
    responseFormat: String,
    instructions: String?,
    modelId: String?
) async throws -> Response {
    let voxModelId: String? = (modelId?.contains("VoxCPM2") == true) ? modelId : nil
    let model = try await modelCache.load(modelId: voxModelId)
    let sampleRate = model.sampleRate
    let sentences = splitIntoSentences(input)
    let contentType = (responseFormat == "wav") ? "audio/wav" : "audio/pcm"
    let format = responseFormat

    return Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init { writer in
            do {
                if format == "wav" {
                    try await writer.write(
                        ByteBuffer(bytes: streamingWAVHeader(sampleRate: sampleRate)))
                }
                for sentence in sentences {
                    let samples = try await model.generateVoxCPM2(
                        text: sentence,
                        language: "english",
                        instruct: instructions)
                    guard !samples.isEmpty else { continue }
                    try await writer.write(ByteBuffer(bytes: float32ToPCM16LE(samples)))
                }
                try await writer.finish(nil)
            } catch {
                try? await writer.finish(nil)
                throw error
            }
        })
}

/// Sentence segmentation for long-form VoxCPM2 synthesis. Splits on
/// `.`/`!`/`?` followed by whitespace, plus paragraph breaks.
///
/// Intentionally simple — abbreviations ("Dr.", "U.S.") cause an extra split
/// which only affects prosody at sentence boundaries, not correctness. If
/// that becomes audible, upgrade to NSLinguisticTagger sentence enumeration.
func splitIntoSentences(_ text: String) -> [String] {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return [] }
    var out: [String] = []
    var current = ""
    let scalars = Array(trimmed)
    var i = 0
    while i < scalars.count {
        let c = scalars[i]
        current.append(c)
        let isTerminal = (c == "." || c == "!" || c == "?" || c == "\n")
        let nextIsSpaceOrEnd = (i + 1 >= scalars.count) || scalars[i + 1].isWhitespace
        if isTerminal && nextIsSpaceOrEnd {
            let segment = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !segment.isEmpty { out.append(segment) }
            current = ""
            while i + 1 < scalars.count && scalars[i + 1].isWhitespace { i += 1 }
        }
        i += 1
    }
    let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
    if !tail.isEmpty { out.append(tail) }
    return out.isEmpty ? [trimmed] : out
}

// MARK: - Streaming WAV header

/// 44-byte RIFF/WAV header advertising "size unknown" via 0x7FFFFFFF for the
/// file and data chunk sizes. Standard streaming-WAV convention: clients stop
/// reading when the connection closes rather than trying to seek to the end.
func streamingWAVHeader(sampleRate: Int) -> [UInt8] {
    var out: [UInt8] = []
    out.reserveCapacity(44)
    func u32(_ v: UInt32) {
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 24) & 0xFF))
    }
    func u16(_ v: UInt16) {
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
    }
    out.append(contentsOf: Array("RIFF".utf8))
    u32(0x7FFF_FFFF)
    out.append(contentsOf: Array("WAVE".utf8))
    out.append(contentsOf: Array("fmt ".utf8))
    u32(16)
    u16(1)
    u16(1)
    u32(UInt32(sampleRate))
    u32(UInt32(sampleRate * 2))
    u16(2)
    u16(16)
    out.append(contentsOf: Array("data".utf8))
    u32(0x7FFF_FFFF)
    return out
}

// MARK: - Model Variant Resolution

/// Maps the OpenAI `model` field to a HuggingFace model ID when it's a
/// recognized VoxCPM2 variant. Anything else returns nil and the engine uses
/// its built-in default.
func resolveVariantModelId(_ raw: String) -> String? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return nil }
    if trimmed.contains("VoxCPM2") { return trimmed }
    return nil
}

// MARK: - Float32 → PCM16 LE

/// Converts Float32 [-1, 1] audio samples to 16-bit little-endian PCM bytes.
/// Clamps before scaling so out-of-range floats don't wrap as Int16 saturation.
func float32ToPCM16LE(_ samples: [Float]) -> Data {
    var out = Data(capacity: samples.count * 2)
    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        let scaled = clamped < 0 ? clamped * 32768 : clamped * 32767
        let i16 = Int16(scaled.rounded())
        out.append(UInt8(truncatingIfNeeded: UInt16(bitPattern: i16) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: UInt16(bitPattern: i16) >> 8))
    }
    return out
}
