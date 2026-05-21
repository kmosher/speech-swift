import CryptoKit
import Foundation
import Hummingbird
import NIOCore
import Qwen3TTS
import CosyVoiceTTS

/// Qwen3-TTS' default sampling (`eosLogitBias: 0.0`) emits the end-of-speech
/// token a frame or two early on plenty of common phrases, so the tail of
/// the last syllable gets clipped. Negative bias pushes EOS later. -2.0 is
/// the default we picked empirically: meaningfully reduces tail clipping
/// without inducing rambling or stuttered trailing silence. Caller can
/// override per-request with `eos_logit_bias` in the OpenAI JSON body.
private let defaultEosLogitBias: Float = -2.0

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

        let samples: [Float]
        let sampleRate: Int

        let eosBias: Float = (json["eos_logit_bias"] as? Double).map(Float.init) ?? defaultEosLogitBias

        switch selector.engine {
        case .qwen3:
            let model = try await modelCache.loadQwen3(modelId: modelId)
            var sampling = SamplingConfig.default
            sampling.eosLogitBias = eosBias
            samples = model.synthesize(
                text: input,
                language: "english",
                speaker: selector.speaker,
                sampling: sampling)
            sampleRate = model.sampleRate
        case .cosyvoice:
            let model = try await modelCache.loadCosyVoice(modelId: modelId)
            samples = model.synthesize(text: input, language: "english")
            sampleRate = 24000
        }

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
            // Raw 16-bit little-endian PCM at the model's native rate.
            // Caller is responsible for knowing the rate (advertised via the
            // standard Kokoro/Qwen3 24kHz convention).
            let pcm = float32ToPCM16LE(samples)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/pcm"],
                body: .init(byteBuffer: .init(data: pcm)))
        }
    }
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

/// Built-in Qwen3-TTS Base-model speakers. Used both for bare-name passthrough
/// and as the bucket that `claude_*` / `blend_*` per-session IDs hash into.
/// The CustomVoice variant has 9 speakers — once it's wired up as a swappable
/// model, this list extends. For now the 4 Base speakers give per-Claude
/// audible distinction even though only `ryan` is true US English.
let qwen3BaseSpeakers: [String] = ["ryan", "vivian", "ono_anna", "sohee"]

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
    // built-in speaker pool. Stable per ID across restarts.
    if trimmed.hasPrefix("claude_") || trimmed.hasPrefix("blend_") {
        let digest = Array(SHA256.hash(data: Data(trimmed.utf8)))
        let firstByte = Int(digest[0])
        let speaker = qwen3BaseSpeakers[firstByte % qwen3BaseSpeakers.count]
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

    // Bare Qwen3 speaker names (the documented English/Chinese/Japanese/Korean set).
    if Set(qwen3BaseSpeakers).contains(trimmed) {
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
