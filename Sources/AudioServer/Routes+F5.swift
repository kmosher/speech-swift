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
        // Do NOT trim the ref audio. F5 aligns the reference audio against the
        // reference *transcript* (clone_ref_text / registry ref.txt); trimming
        // the audio without trimming the text to match desyncs that alignment,
        // which makes F5's duration predictor wildly over-estimate and fill the
        // excess with noise (audible clicking / "hoofbeats"). Registry refs are
        // already short (~18s), so the full clip is what the transcript covers.
        let samples = try decodeWAVData(data, targetSampleRate: F5TTS.sampleRate)
        let arr = MLXArray(samples)
        entries[ref] = arr
        return arr
    }
}

/// Resolve the reference F5 should clone from. If an `f5_ref.wav` + `f5_ref.txt`
/// pair sits beside the given registry `ref.wav`, prefer it (a clean, aligned,
/// short clip purpose-built for F5); otherwise return the original ref and its
/// supplied transcript. Only registry-style local `…/ref.wav` paths are
/// rewritten — ad-hoc `clone_ref` paths/URLs pass through untouched.
func f5PreferredRef(forRef ref: String, fallbackText: String?) -> (String, String?) {
    guard ref.hasSuffix("/ref.wav") else { return (ref, fallbackText) }
    let dir = String(ref.dropLast("ref.wav".count))  // retains trailing slash
    let f5Wav = dir + "f5_ref.wav"
    let f5Txt = dir + "f5_ref.txt"
    let fm = FileManager.default
    guard fm.fileExists(atPath: f5Wav), fm.fileExists(atPath: f5Txt),
          let text = try? String(contentsOfFile: f5Txt, encoding: .utf8) else {
        return (ref, fallbackText)
    }
    return (f5Wav, text.trimmingCharacters(in: .whitespacesAndNewlines))
}

/// F5 voice clone. Emits a streaming WAV/PCM body, but unlike handleVoxCPM2Clone
/// it synthesizes the whole input in a single F5 pass (F5 generates continuous
/// multi-sentence speech; per-sentence batching would add join artifacts). F5
/// has no style/`instruct` parameter, so `instructions` is not threaded here.
func handleF5Clone(
    input: String,
    cloneRef: String,
    cloneRefText: String?,
    responseFormat: String
) async throws -> Response {
    // Prefer an F5-specific reference pair when one sits next to the registry
    // ref. `f5_ref.wav` + `f5_ref.txt` (built by tts-bench/build_f5_refs.py) are
    // a silence-trimmed ~10s 24kHz clip and a transcript generated FROM that
    // clip, so the audio and text stay aligned and short — which F5 needs and
    // the VoxCPM2-curated ref.wav/ref.txt don't guarantee. Falls back to the
    // supplied ref when no F5 pair exists (e.g. an ad-hoc clone_ref URL).
    let (resolvedRef, resolvedRefText) = f5PreferredRef(forRef: cloneRef, fallbackText: cloneRefText)

    let model: F5TTS
    let refAudio: MLXArray
    do {
        model = try await f5Cache.get()
        refAudio = try await f5RefCache.load(ref: resolvedRef)
    } catch {
        FileHandle.standardError.write(Data("[f5] load failed: \(error)\n".utf8))
        return errorResponse("F5 load failed: \(error)", status: .internalServerError)
    }
    let refText = resolvedRefText ?? ""
    // F5 is 24kHz native, which already equals the OpenAI PCM wire rate, so
    // neither WAV nor PCM needs resampling on the way out.
    let sampleRate = F5TTS.sampleRate
    let contentType = (responseFormat == "wav") ? "audio/wav" : "audio/pcm"
    let format = responseFormat

    // Synthesize the WHOLE input in one F5 pass — do NOT split into sentences
    // the way the VoxCPM2 path does. VoxCPM2 needs per-sentence batching for
    // stability; F5 is trained to generate continuous multi-sentence speech and
    // re-conditioning per sentence instead produces audible artifacts at every
    // join (a "clatter" between fragments). One pass also lets the duration
    // estimate see the full text. (TTFB cost: the whole clip is synthesized
    // before the first bytes go out — acceptable for these short TTS inputs.)
    return Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init { writer in
            do {
                if format == "wav" {
                    try await writer.write(
                        ByteBuffer(bytes: streamingWAVHeader(sampleRate: sampleRate)))
                }
                let out = try await model.generate(
                    text: input,
                    referenceAudio: refAudio,
                    referenceAudioText: refText)
                let samples = out.asArray(Float.self)
                if !samples.isEmpty {
                    try await writer.write(ByteBuffer(bytes: float32ToPCM16LE(samples)))
                }
                try await writer.finish(nil)
            } catch {
                try? await writer.finish(nil)
                throw error
            }
        })
}
