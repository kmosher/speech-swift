import Foundation
import Hummingbird
import NIOCore
import Qwen3TTS
import CosyVoiceTTS

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

        let samples: [Float]
        let sampleRate: Int

        switch selector.engine {
        case .qwen3:
            let model = try await state.loadTTS()
            samples = model.synthesize(text: input, language: "english", speaker: selector.speaker)
            sampleRate = model.sampleRate
        case .cosyvoice:
            let model = try await state.loadCosyVoice()
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

/// Parses the OpenAI `voice` field into engine + speaker.
///
/// Accepted shapes:
/// - `"qwen3:<speaker>"` or `"qwen3/<speaker>"` → Qwen3-TTS with that speaker
/// - `"cosyvoice"` or `"cosyvoice:..."` → CosyVoice (speaker ignored for now)
/// - bare Qwen3 speaker names (`"ryan"`, `"vivian"`, `"sohee"`, `"ono_anna"`)
///   → Qwen3 with that speaker
/// - OpenAI's stock voices (`"alloy"`, `"echo"`, etc.) → Qwen3 with default
///   speaker (so OpenAI-targeted clients work without modification)
/// - anything else → Qwen3 with default speaker
func parseVoiceSelector(_ raw: String) -> VoiceSelector {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if trimmed.isEmpty {
        return VoiceSelector(engine: .qwen3, speaker: nil)
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
    let qwen3Speakers: Set<String> = ["ryan", "vivian", "ono_anna", "sohee"]
    if qwen3Speakers.contains(trimmed) {
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
