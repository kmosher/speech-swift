import AudioCommon
import F5TTS
import Foundation
import Hummingbird
import MLX
import NIOCore

// MARK: - F5-TTS engine for /v1/audio/speech
//
// F5-TTS is a flow-matching voice-cloning TTS, 24kHz output. It's the default
// engine for cloned / registry voices on this server; VoxCPM2 stays reachable
// (see the routing precedence in Routes+OpenAI.swift). F5 is materially lighter
// and faster than VoxCPM2 (≈0.4 RTF vs ≈1.2) at comparable clone quality.

/// Default F5 weights repo. F5TTS.fromPretrained pulls model.safetensors +
/// vocab.txt + duration_v2.safetensors from here; the mel vocoder is a separate
/// auto-download (lucasnewman/vocos-mel-24khz-mlx) cached on the model.
let F5DefaultModelId = "lucasnewman/f5-tts-mlx"

/// F5 conditions on the *entire* reference clip inline, so a long ref slows the
/// flow integration and skews the duration estimate. Registry refs can run
/// ~20s; trim to this many seconds — enough to carry timbre, bounded in cost.
private let F5MaxRefSeconds = 12

private let f5Cache = F5Cache()

/// Single resident F5 model. Unlike VoxCPM2 there are no size/precision
/// variants to key on, so one slot suffices.
actor F5Cache {
    private var model: F5TTS?

    func get() async throws -> F5TTS {
        if let model { return model }
        let m = try await F5TTS.fromPretrained(repoId: F5DefaultModelId)
        model = m
        return m
    }

    /// Drop the resident model so its MLX buffers return to the pool for the
    /// idle monitor's clearCache(). Returns 1 if one was loaded, else 0.
    func evict() -> Int {
        let n = (model == nil) ? 0 : 1
        model = nil
        return n
    }
}

/// Release the resident F5 model (idle-monitor hook, mirroring evictTTSModels).
func evictF5Models() async -> Int {
    await f5Cache.evict()
}

/// 24kHz mono reference cache for F5, keyed by source path/URL. F5's mel
/// spectrogram runs at 24kHz and applies no rate correction, so the ref must be
/// resampled to exactly that — otherwise the clone comes out pitch-shifted
/// ("chipmunk"). VoxCPM2's separate ref cache decodes at 16kHz, hence a
/// distinct cache here.
private let f5RefCache = F5RefCache()

actor F5RefCache {
    private var entries: [String: MLXArray] = [:]

    func load(ref: String) async throws -> MLXArray {
        if let cached = entries[ref] { return cached }
        let data: Data
        if ref.hasPrefix("http://") || ref.hasPrefix("https://") {
            guard let url = URL(string: ref) else {
                throw CloneError.invalidRef("Invalid URL: \(ref)")
            }
            let (fetched, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw CloneError.invalidRef("Fetched \(ref): HTTP \(http.statusCode)")
            }
            data = fetched
        } else {
            guard ref.hasPrefix("/") else {
                throw CloneError.invalidRef(
                    "clone_ref must be an absolute path or http(s):// URL (got '\(ref)')")
            }
            data = try Data(contentsOf: URL(fileURLWithPath: ref))
        }
        var samples = try decodeWAVData(data, targetSampleRate: F5TTS.sampleRate)
        let cap = F5MaxRefSeconds * F5TTS.sampleRate
        if samples.count > cap {
            samples = Array(samples[..<cap])
        }
        let arr = MLXArray(samples)
        entries[ref] = arr
        return arr
    }
}

/// F5 voice clone. Mirrors handleVoxCPM2Clone's streaming-WAV shape: split the
/// input into sentences and write each sentence's PCM as it finishes, so
/// multi-sentence replies start playing before the whole thing is synthesized.
/// F5 has no style/`instruct` parameter, so `instructions` is not threaded here.
func handleF5Clone(
    input: String,
    cloneRef: String,
    cloneRefText: String?,
    responseFormat: String
) async throws -> Response {
    let model: F5TTS
    let refAudio: MLXArray
    do {
        model = try await f5Cache.get()
        refAudio = try await f5RefCache.load(ref: cloneRef)
    } catch {
        FileHandle.standardError.write(Data("[f5] load failed: \(error)\n".utf8))
        return errorResponse("F5 load failed: \(error)", status: .internalServerError)
    }
    let refText = cloneRefText ?? ""
    // F5 is 24kHz native, which already equals the OpenAI PCM wire rate, so
    // neither WAV nor PCM needs resampling on the way out.
    let sampleRate = F5TTS.sampleRate
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
                    let out = try await model.generate(
                        text: sentence,
                        referenceAudio: refAudio,
                        referenceAudioText: refText)
                    let samples = out.asArray(Float.self)
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
