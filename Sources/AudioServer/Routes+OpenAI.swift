import CryptoKit
import Foundation
import Hummingbird
import NIOCore
import Qwen3TTS
import CosyVoiceTTS
import VoxCPM2TTS

/// Default EOS-logit bias for Qwen3-TTS, per model variant.
///
/// The Base variants emit EOS a frame or two early on common phrases (tail
/// of the last syllable clips), so we push EOS out with a negative bias.
/// CustomVoice's EOS is calibrated correctly and the same negative bias
/// makes it ramble for 6+ seconds on a 3-second phrase. Empirically:
///   - Base 8-bit + bias=0.0 → clipped tail
///   - Base 8-bit + bias=-2.0 → clean
///   - CustomVoice 4-bit + bias=0.0 → clean
///   - CustomVoice 4-bit + bias=-1.0 → drawn-out
///
/// Caller can always override per-request with `eos_logit_bias`.
private func defaultEosLogitBias(forModelId modelId: String?) -> Float {
    if let modelId, modelId.contains("CustomVoice") { return 0.0 }
    return -2.0
}

// MARK: - OpenAI-Compatible REST Routes
//
// Adds `POST /v1/audio/speech` (and eventually `/v1/audio/transcriptions`)
// in the shape OpenAI's HTTP TTS API documents:
//   https://platform.openai.com/docs/api-reference/audio/createSpeech
//
// Clients that target OpenAI's `/v1/audio/speech` (voicemode, openai-python,
// many integration libs) can point at this server unchanged.
//
// Engine + speaker selection happens entirely through the `voice` field.
// See `parseVoiceSelector(_:)` for the parsing rules.

/// Per-model variant cache. Lets clients select Qwen3 size+precision per
/// request via the OpenAI `model` field without rebuilding the server, and
/// keeps previously-loaded variants resident so A/B testing doesn't pay the
/// download/load cost on every switch. Each variant is a few GB of MLX
/// weights, so callers should know what they're loading.
private let modelCache = VariantModelCache()

actor VariantModelCache {
    private var qwen3: [String: Qwen3TTSModel] = [:]
    private var cosyvoice: [String: CosyVoiceTTSModel] = [:]
    private var voxcpm2: [String: VoxCPM2TTSModel] = [:]

    func loadVoxCPM2(modelId: String?) async throws -> VoxCPM2TTSModel {
        let key = modelId ?? "default"
        if let m = voxcpm2[key] { return m }
        let m: VoxCPM2TTSModel
        if let modelId = modelId {
            m = try await VoxCPM2TTSModel.fromPretrained(modelId: modelId)
        } else {
            m = try await VoxCPM2TTSModel.fromPretrained()
        }
        voxcpm2[key] = m
        return m
    }

    func loadQwen3(modelId: String?) async throws -> Qwen3TTSModel {
        let key = modelId ?? "default"
        if let m = qwen3[key] { return m }
        let m: Qwen3TTSModel
        if let modelId = modelId {
            m = try await Qwen3TTSModel.fromPretrained(modelId: modelId)
        } else {
            m = try await Qwen3TTSModel.fromPretrained()
        }
        qwen3[key] = m
        return m
    }

    func loadCosyVoice(modelId: String?) async throws -> CosyVoiceTTSModel {
        let key = modelId ?? "default"
        if let m = cosyvoice[key] { return m }
        let m: CosyVoiceTTSModel
        if let modelId = modelId {
            m = try await CosyVoiceTTSModel.fromPretrained(modelId: modelId)
        } else {
            m = try await CosyVoiceTTSModel.fromPretrained()
        }
        cosyvoice[key] = m
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

        let voiceRaw = (json["voice"] as? String) ?? "qwen3:ryan"
        let selector = parseVoiceSelector(voiceRaw)
        let responseFormat = ((json["response_format"] as? String) ?? "wav").lowercased()
        guard responseFormat == "wav" || responseFormat == "pcm" else {
            // OpenAI also documents mp3/opus/aac/flac. Synthesizing them would
            // need ffmpeg conversion of the raw float samples — out of scope
            // for the first cut.
            return errorResponse(
                "'response_format' must be 'wav' or 'pcm' (mp3/opus/aac/flac not supported yet)",
                status: .badRequest)
        }

        // `model` field is the OpenAI hook for selecting model size/precision.
        // We treat it as an HF model ID when it matches a recognized prefix;
        // anything else (incl. OpenAI's "tts-1"/"tts-1-hd") falls through to
        // the engine's default. Variants are cached after first load.
        let modelId = (json["model"] as? String).flatMap(resolveVariantModelId)

        let eosBias: Float = (json["eos_logit_bias"] as? Double).map(Float.init)
            ?? defaultEosLogitBias(forModelId: modelId)

        // OpenAI's gpt-4o-mini-tts API uses `instructions` for natural-language
        // style control ("speak excitedly", "in a low whisper", etc.). Qwen3
        // CustomVoice supports the exact same idea via its `instruct` arg, so
        // we expose it under OpenAI's field name. Base model ignores it.
        // Empty string is treated as "no override" — falls back to the engine's
        // default ("Speak naturally." for CustomVoice).
        let instructions: String? = (json["instructions"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }

        // Voice cloning hook (custom extension to OpenAI's shape). `clone_ref`
        // is either an absolute filesystem path or an https:// URL pointing at
        // a WAV/audio file. When set we route to a clone-capable engine, ignore
        // the speaker / bucketing logic, and synthesize in the reference voice.
        //
        // Engine selection (`clone_engine`, default "voxcpm2"):
        //   - "voxcpm2": VoxCPM2 (default, higher fidelity). Best results need
        //     `clone_ref_text` — the transcript of the reference clip — which
        //     enables VoxCPM2's in-context prompt path. Without it we fall back
        //     to VoxCPM2's audio-only ref path (lower fidelity).
        //   - "qwen3":   Qwen3 0.6B Base + ECAPA-TDNN speaker embedding. No
        //     transcript needed; the older, lower-quality path.
        // Reference audio is cached by (path, sample-rate) so repeat calls skip
        // the decode/fetch.
        let cloneRef: String? = (json["clone_ref"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        let cloneRefText: String? = (json["clone_ref_text"] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        let cloneEngine = ((json["clone_engine"] as? String) ?? "voxcpm2")
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        // Registry-driven cloning: if no explicit clone_ref was given, the
        // `voice` field can still resolve to a pre-curated registry entry.
        // - Bare id ("ed_irish_m") → exact lookup.
        // - "claude_<hex>" / "blend_<hex>" → SHA-256 hashed across the sorted
        //   registry. Same session id → same voice across restarts.
        // Empty registry falls through to legacy Qwen3 bucketing in
        // parseVoiceSelector.
        let registryEntry: VoiceEntry? = {
            if cloneRef != nil { return nil }  // explicit clone wins
            if voiceRegistry.isEmpty { return nil }
            let raw = voiceRaw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if raw.hasPrefix("claude_") || raw.hasPrefix("blend_") {
                return voiceRegistry.hashedLookup(sessionId: raw)
            }
            return voiceRegistry.lookup(id: raw)
        }()

        if let cloneRef {
            if cloneEngine == "qwen3" {
                return try await handleQwen3Clone(
                    input: input,
                    cloneRef: cloneRef,
                    responseFormat: responseFormat,
                    modelId: modelId)
            }
            return try await handleVoxCPM2Clone(
                input: input,
                cloneRef: cloneRef,
                cloneRefText: cloneRefText,
                responseFormat: responseFormat,
                instructions: instructions,
                modelId: modelId)
        }
        if let entry = registryEntry {
            return try await handleVoxCPM2Clone(
                input: input,
                cloneRef: entry.refPath,
                cloneRefText: entry.refText,
                responseFormat: responseFormat,
                instructions: instructions,
                modelId: modelId)
        }

        switch selector.engine {
        case .qwen3:
            let model = try await modelCache.loadQwen3(modelId: modelId)
            // `let` (not `var`) so the streaming-body closure can capture it
            // without tripping Swift 6's SendableClosureCaptures rule.
            let sampling: SamplingConfig = {
                var s = SamplingConfig.default
                s.eosLogitBias = eosBias
                return s
            }()
            let sampleRate = model.sampleRate
            let speaker = selector.speaker
            let instruct = instructions

            let contentType = (responseFormat == "wav") ? "audio/wav" : "audio/pcm"
            // Streaming response body: write chunks as the model emits them.
            // For WAV, the first chunk is a streaming RIFF header with
            // 0x7FFFFFFF size sentinels — clients (afplay, ffplay, AVPlayer,
            // browsers via MediaSource) treat that as "play until connection
            // closes" instead of trying to seek to the end.
            return Response(
                status: .ok,
                headers: [.contentType: contentType],
                body: .init { writer in
                    do {
                        if responseFormat == "wav" {
                            try await writer.write(
                                ByteBuffer(bytes: streamingWAVHeader(sampleRate: sampleRate)))
                        }
                        let stream = model.synthesizeStream(
                            text: input,
                            language: "english",
                            speaker: speaker,
                            instruct: instruct,
                            sampling: sampling)
                        for try await chunk in stream {
                            guard !chunk.samples.isEmpty else { continue }
                            let pcm = float32ToPCM16LE(chunk.samples)
                            try await writer.write(ByteBuffer(bytes: pcm))
                        }
                        try await writer.finish(nil)
                    } catch {
                        // Once we've sent headers we can't switch to a 4xx/5xx,
                        // so the best we can do is end the stream early. Body
                        // consumers will see the truncation; the error path is
                        // visible in server logs.
                        try? await writer.finish(nil)
                        throw error
                    }
                })

        case .cosyvoice:
            // CosyVoice synth is still one-shot here; streaming variant is
            // available upstream but we haven't wired it yet. Same accumulate-
            // then-respond shape we had before.
            let model = try await modelCache.loadCosyVoice(modelId: modelId)
            let samples = model.synthesize(text: input, language: "english")
            let sampleRate = 24000
            if samples.isEmpty {
                return errorResponse("Synthesis produced no audio", status: .internalServerError)
            }
            if responseFormat == "wav" {
                let wav = try encodeWAV(samples: samples, sampleRate: sampleRate)
                return Response(
                    status: .ok,
                    headers: [.contentType: "audio/wav"],
                    body: .init(byteBuffer: .init(data: wav)))
            } else {
                let pcm = float32ToPCM16LE(samples)
                return Response(
                    status: .ok,
                    headers: [.contentType: "audio/pcm"],
                    body: .init(byteBuffer: .init(data: pcm)))
            }
        }
    }
}

// MARK: - Voice Cloning

/// Cache of decoded reference audio + sample rate keyed by source path/URL.
/// The model's own internal cache memoizes the speaker embedding it derives
/// from the audio; this cache memoizes the audio decode + (when remote) the
/// HTTP fetch so repeated /v1/audio/speech calls don't redownload.
private let refAudioCache = RefAudioCache()

actor RefAudioCache {
    // Keyed by "<ref>@<rate>" so the same source decoded at two rates (Qwen3
    // wants 24kHz, VoxCPM2 wants 16kHz) doesn't collide.
    private var entries: [String: (samples: [Float], sampleRate: Int)] = [:]

    func load(ref: String, targetSampleRate: Int) async throws -> (samples: [Float], sampleRate: Int) {
        let cacheKey = "\(ref)@\(targetSampleRate)"
        if let cached = entries[cacheKey] { return cached }
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
            // Reject anything that doesn't look like an absolute path. Relative
            // paths would resolve against the server's cwd, which is rarely
            // what callers expect.
            guard ref.hasPrefix("/") else {
                throw CloneError.invalidRef(
                    "clone_ref must be an absolute path or http(s):// URL (got '\(ref)')")
            }
            data = try Data(contentsOf: URL(fileURLWithPath: ref))
        }
        // decodeWAVData lives in AudioServer.swift; it writes to a temp file
        // then uses AudioFileLoader, which handles WAV/M4A/MP3/etc via
        // AVAudioFile.
        let samples = try decodeWAVData(data, targetSampleRate: targetSampleRate)
        let entry = (samples: samples, sampleRate: targetSampleRate)
        entries[cacheKey] = entry
        return entry
    }
}

enum CloneError: Error, LocalizedError {
    case invalidRef(String)
    case modelHasNoEncoder

    var errorDescription: String? {
        switch self {
        case .invalidRef(let msg): return msg
        case .modelHasNoEncoder:
            return "Selected model has no speaker encoder (cloning requires a Base variant, not CustomVoice)"
        }
    }
}

/// VoxCPM2 voice clone (the default, higher-fidelity path). VoxCPM2 is 2B /
/// 48kHz and designed for zero-shot cloning: given the reference clip *and*
/// its transcript it uses an in-context "prompt" (continue-this-voice) which
/// is materially better than embedding-only cloning. Without the transcript
/// we fall back to the audio-only `refAudio` conditioning.
///
/// Long inputs are split into sentences and synthesized per-sentence — VoxCPM2
/// destabilizes (speed-up, buzzing) on long single calls per the upstream
/// usage guide, and per-sentence streaming also cuts TTFB on multi-sentence
/// inputs.
///
/// Reference audio is decoded at 16kHz (VoxCPM2's encoder rate). The decoded
/// float buffer is cached in `refAudioCache`; the model still re-encodes the
/// VAE features per call. That cost is small relative to the LM autoregress,
/// so a deeper cache (encoded-prompt-tensor keyed by ref path) is deferred.
private func handleVoxCPM2Clone(
    input: String,
    cloneRef: String,
    cloneRefText: String?,
    responseFormat: String,
    instructions: String?,
    modelId: String?
) async throws -> Response {
    // Honor an explicit VoxCPM2 model id; otherwise use the engine default.
    let voxModelId: String? = (modelId?.contains("VoxCPM2") == true) ? modelId : nil
    let model = try await modelCache.loadVoxCPM2(modelId: voxModelId)
    let (refSamples, _) = try await refAudioCache.load(ref: cloneRef, targetSampleRate: 16000)
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
                    let pcm = float32ToPCM16LE(samples)
                    try await writer.write(ByteBuffer(bytes: pcm))
                }
                try await writer.finish(nil)
            } catch {
                try? await writer.finish(nil)
                throw error
            }
        })
}

/// Sentence segmentation for long-form VoxCPM2 synthesis. Splits on
/// `.`/`!`/`?` followed by whitespace, plus paragraph breaks. Single-token
/// inputs and inputs without terminal punctuation come back as a single
/// segment — VoxCPM2 handles short clean inputs fine.
///
/// This is intentionally simple. We don't try to detect abbreviations
/// ("Dr.", "U.S.") — at worst they cause an extra split which only affects
/// prosody at sentence boundaries, not correctness. If that becomes audible,
/// upgrade to NSLinguisticTagger sentence enumeration.
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
            // Skip following whitespace
            while i + 1 < scalars.count && scalars[i + 1].isWhitespace { i += 1 }
        }
        i += 1
    }
    let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
    if !tail.isEmpty { out.append(tail) }
    return out.isEmpty ? [trimmed] : out
}

/// Qwen3 0.6B Base voice clone (legacy path, `clone_engine:"qwen3"`). Base-only
/// — CustomVoice doesn't ship the ECAPA-TDNN encoder, so a CustomVoice/unknown
/// `model` is silently forced to 0.6B Base 8-bit. Embedding-only; no transcript.
private func handleQwen3Clone(
    input: String,
    cloneRef: String,
    responseFormat: String,
    modelId: String?
) async throws -> Response {
    let baseModelId: String
    if let modelId, modelId.contains("Qwen3-TTS-") && !modelId.contains("CustomVoice") {
        baseModelId = modelId
    } else {
        baseModelId = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit"
    }
    let model = try await modelCache.loadQwen3(modelId: baseModelId)
    let (refSamples, refRate) = try await refAudioCache.load(ref: cloneRef, targetSampleRate: 24000)
    let samples = model.synthesizeWithVoiceClone(
        text: input,
        referenceAudio: refSamples,
        referenceSampleRate: refRate,
        language: "english")
    if samples.isEmpty {
        return errorResponse("Cloned synthesis produced no audio", status: .internalServerError)
    }
    return try audioResponse(samples: samples, sampleRate: model.sampleRate, responseFormat: responseFormat)
}

/// Shared one-shot (non-streaming) audio response: WAV-encode or raw PCM16LE.
private func audioResponse(samples: [Float], sampleRate: Int, responseFormat: String) throws -> Response {
    if responseFormat == "wav" {
        let wav = try encodeWAV(samples: samples, sampleRate: sampleRate)
        return Response(
            status: .ok,
            headers: [.contentType: "audio/wav"],
            body: .init(byteBuffer: .init(data: wav)))
    } else {
        let pcm = float32ToPCM16LE(samples)
        return Response(
            status: .ok,
            headers: [.contentType: "audio/pcm"],
            body: .init(byteBuffer: .init(data: pcm)))
    }
}

// MARK: - Streaming WAV header

/// 44-byte RIFF/WAV header advertising "size unknown" via the 0x7FFFFFFF
/// sentinel for the file and data chunk sizes. Standard streaming-WAV
/// convention: clients stop reading when the connection closes rather than
/// trying to seek to the declared end. Mirrors the helper macos-speech-server
/// uses for its Kokoro-on-FluidAudio streaming response.
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
    u32(0x7FFF_FFFF)  // unknown file size
    out.append(contentsOf: Array("WAVE".utf8))
    out.append(contentsOf: Array("fmt ".utf8))
    u32(16)                              // PCM fmt chunk size
    u16(1)                               // PCM format
    u16(1)                               // mono
    u32(UInt32(sampleRate))
    u32(UInt32(sampleRate * 2))          // byte rate (16-bit mono)
    u16(2)                               // block align
    u16(16)                              // bits per sample
    out.append(contentsOf: Array("data".utf8))
    u32(0x7FFF_FFFF)                     // unknown data size
    return out
}

// MARK: - Model Variant Resolution

/// Maps the OpenAI `model` field to a HuggingFace model ID when it's a
/// recognized variant prefix, otherwise returns nil (engine uses its built-in
/// default). Currently recognizes the aufklarer-published Qwen3-TTS and
/// CosyVoice MLX variants. Bare prefix forms ("qwen3-0.6b-8bit" etc.) are a
/// future shorthand; today we accept the full HF path.
func resolveVariantModelId(_ raw: String) -> String? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return nil }
    if trimmed.hasPrefix("aufklarer/Qwen3-TTS-") { return trimmed }
    if trimmed.hasPrefix("aufklarer/CosyVoice") { return trimmed }
    return nil
}

// MARK: - Voice → Engine + Speaker Parsing

enum SynthesisEngine: Sendable {
    case qwen3
    case cosyvoice
}

struct VoiceSelector: Sendable {
    let engine: SynthesisEngine
    /// Speaker name within the engine's built-in set (e.g. "ryan", "vivian").
    /// nil → engine picks its own default for the language.
    let speaker: String?
}

/// Speaker bucket that `claude_*` / `blend_*` per-session IDs hash into.
/// Restricted to the English-native CustomVoice speakers so per-Claude voices
/// don't land on Mandarin/Japanese/Korean-accented English mid-sentence.
/// CustomVoice has 9 total (also: vivian, ono_anna, sohee, eric+dylan dialects,
/// uncle_fu); extend this list if you want more variety and accept the
/// language-mismatch artifacts.
/// Note: the Base model variants have empty spk_id and silently ignore the
/// speaker parameter — only CustomVoice actually honors speaker selection.
let qwen3BucketSpeakers: [String] = ["ryan", "serena", "aiden"]

/// Bare names recognized by parseVoiceSelector outside the bucket. Includes
/// all 9 CustomVoice speakers + the Base "speakers" (which are no-ops on
/// Base but still accepted for forward-compat with future variants).
let qwen3KnownSpeakers: Set<String> = [
    "ryan", "serena", "aiden", "vivian", "ono_anna", "sohee",
    "eric", "dylan", "uncle_fu",
]

/// Parses the OpenAI `voice` field into engine + speaker.
///
/// Accepted shapes:
/// - `"qwen3:<speaker>"` or `"qwen3/<speaker>"` → Qwen3-TTS with that speaker
/// - `"cosyvoice"` or `"cosyvoice:..."` → CosyVoice (speaker ignored for now)
/// - bare Qwen3 speaker names (`"ryan"`, `"vivian"`, `"sohee"`, `"ono_anna"`)
///   → Qwen3 with that speaker
/// - `"claude_<hex>"` / `"blend_<hex>"` → deterministically hashed into the
///   Qwen3 speaker pool. Same ID always picks the same speaker, so each
///   Claude session gets a stable per-session voice without coordination.
/// - OpenAI's stock voices (`"alloy"`, `"echo"`, etc.) → Qwen3 with default
///   speaker (so OpenAI-targeted clients work without modification)
/// - anything else → Qwen3 with default speaker
func parseVoiceSelector(_ raw: String) -> VoiceSelector {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if trimmed.isEmpty {
        return VoiceSelector(engine: .qwen3, speaker: nil)
    }

    // Per-session bucketing: any claude_/blend_ prefix hashes into the
    // English-native speaker pool. Stable per ID across restarts.
    if trimmed.hasPrefix("claude_") || trimmed.hasPrefix("blend_") {
        let digest = Array(SHA256.hash(data: Data(trimmed.utf8)))
        let firstByte = Int(digest[0])
        let speaker = qwen3BucketSpeakers[firstByte % qwen3BucketSpeakers.count]
        return VoiceSelector(engine: .qwen3, speaker: speaker)
    }

    // engine:speaker / engine/speaker prefix form.
    for sep in [":", "/"] {
        if let idx = trimmed.firstIndex(of: Character(sep)) {
            let enginePart = String(trimmed[..<idx])
            let speakerPart = String(trimmed[trimmed.index(after: idx)...])
            switch enginePart {
            case "qwen3":
                return VoiceSelector(
                    engine: .qwen3,
                    speaker: speakerPart.isEmpty ? nil : speakerPart)
            case "cosyvoice":
                return VoiceSelector(engine: .cosyvoice, speaker: nil)
            default:
                break  // fall through to bare-name handling
            }
        }
    }

    // Bare Qwen3 speaker names — accepted whether or not the active variant
    // actually honors them. (Base variants ignore the speaker; CustomVoice
    // uses it.)
    if qwen3KnownSpeakers.contains(trimmed) {
        return VoiceSelector(engine: .qwen3, speaker: trimmed)
    }

    // "cosyvoice" with no speaker.
    if trimmed == "cosyvoice" {
        return VoiceSelector(engine: .cosyvoice, speaker: nil)
    }

    // OpenAI stock voices (alloy/echo/fable/onyx/nova/shimmer) and any unknown
    // string: fall through to Qwen3 default. Lets OpenAI-targeted clients
    // work without knowing about Qwen3 speaker names.
    return VoiceSelector(engine: .qwen3, speaker: nil)
}

// MARK: - Float32 → PCM16 LE

/// Converts Float32 [-1, 1] audio samples to 16-bit little-endian PCM bytes.
/// Mirrors the conversion macos-speech-server's Kokoro service uses on its
/// streaming PCM path. Clamps before scaling so out-of-range floats don't
/// wrap around as Int16 saturation.
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
