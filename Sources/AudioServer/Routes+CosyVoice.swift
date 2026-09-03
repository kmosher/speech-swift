import AudioCommon
import CosyVoiceTTS
import Foundation
import Hummingbird
import NIOCore

// MARK: - CosyVoice on /v1/audio/speech
//
// The third cloning engine on this route, alongside F5 (fast, 24kHz) and
// VoxCPM2 (slow, 48kHz). CosyVoice clones differently from both: rather than
// conditioning on the reference waveform, it takes a 192-dimensional CAM++
// speaker embedding, so a reference is reduced to a vector once and that
// vector is what steers synthesis.
//
// That indirection is why this file exists rather than another arm in
// Routes+OpenAI: cloning here needs a second model (the CAM++ extractor) and
// a second cache keyed by reference path, since re-embedding a reference on
// every request would cost more than the synthesis it feeds.

/// CosyVoice weights are a couple of GB of MLX; keep variants resident across
/// requests the way the VoxCPM2 cache does.
private let cosyCache = CosyVoiceCache()

actor CosyVoiceCache {
    private var entries: [String: CosyVoiceTTSModel] = [:]
    private var speaker: CamPlusPlusSpeaker?

    func load(modelId: String?) async throws -> CosyVoiceTTSModel {
        let key = modelId ?? "default"
        if let m = entries[key] { return m }
        let m: CosyVoiceTTSModel
        if let modelId {
            m = try await CosyVoiceTTSModel.fromPretrained(modelId: modelId)
        } else {
            m = try await CosyVoiceTTSModel.fromPretrained()
        }
        entries[key] = m
        return m
    }

    /// The CAM++ extractor is shared across every voice — it is a CoreML model
    /// of a few MB whose only job is waveform → 192-dim embedding.
    func loadSpeaker() async throws -> CamPlusPlusSpeaker {
        if let speaker { return speaker }
        let s = try await CamPlusPlusSpeaker.fromPretrained()
        speaker = s
        return s
    }

    func evict() -> Int {
        let n = entries.count
        entries.removeAll()
        speaker = nil
        return n
    }
}

/// Release resident CosyVoice models. Mirrors `evictTTSModels` so the idle
/// monitor drops every engine, not just the one it was written for.
func evictCosyVoiceModels() async -> Int {
    await cosyCache.evict()
}

/// Speaker embeddings keyed by reference path.
///
/// A registry voice's `ref.wav` never changes, and CAM++ is deterministic, so
/// the embedding for a given reference is computed once for the life of the
/// process. This matters more than it looks: the voice lab renders the whole
/// registry in a batch, which without a cache would run the extractor 87 times
/// over the same handful of references.
private let speakerCache = SpeakerEmbeddingCache()

actor SpeakerEmbeddingCache {
    private var entries: [String: [Float]] = [:]

    func embedding(for ref: String, samples: [Float],
                   speaker: CamPlusPlusSpeaker) throws -> [Float] {
        if let e = entries[ref] { return e }
        // CAM++ wants 16kHz; the shared reference loader hands back 24kHz
        // because that is what the synthesis engines take.
        let mono16k = AudioFileLoader.resample(samples, from: 24_000, to: 16_000)
        let e = try speaker.embed(audio: mono16k, sampleRate: 16_000)
        entries[ref] = e
        return e
    }
}

/// Synthesize `input` in the voice of `cloneRef` using CosyVoice.
///
/// Takes no reference transcript, unlike the F5 and VoxCPM2 handlers: the
/// embedding path conditions purely on the speaker vector, so `ref.txt` has
/// nowhere to go. CosyVoice's transcript-driven zero-shot mode is a separate
/// path we do not expose. Worth knowing when comparing engines — this is a
/// structurally different clone, not the same one with other weights.
func handleCosyVoiceClone(
    input: String,
    cloneRef: String,
    responseFormat: String,
    modelId: String?
) async throws -> Response {
    let model = try await cosyCache.load(modelId: modelId)
    let refSamples = try await refAudioCache.load(ref: cloneRef)
    let speaker = try await cosyCache.loadSpeaker()
    let embedding = try await speakerCache.embedding(
        for: cloneRef, samples: refSamples, speaker: speaker)

    let sampleRate = model.sampleRate
    let samples = model.synthesize(
        text: input, language: "english", speakerEmbedding: embedding)

    let wireSampleRate = (responseFormat == "pcm") ? PCMWireSampleRate : sampleRate
    let wire = (wireSampleRate == sampleRate)
        ? samples
        : AudioFileLoader.resample(samples, from: sampleRate, to: wireSampleRate)

    let contentType = (responseFormat == "wav") ? "audio/wav" : "audio/pcm"
    let body: Data = (responseFormat == "wav")
        ? try encodeWAV(samples: wire, sampleRate: wireSampleRate)
        : float32ToPCM16LE(wire)
    return Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init(byteBuffer: .init(data: body)))
}
