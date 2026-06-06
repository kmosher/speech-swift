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

/// Read a Double from the environment, falling back to `dflt` when unset/unparseable.
private func envDouble(_ name: String, _ dflt: Double) -> Double {
    guard let s = ProcessInfo.processInfo.environment[name], let v = Double(s) else { return dflt }
    return v
}

/// Read an Int from the environment, falling back to `dflt` when unset/unparseable.
private func envInt(_ name: String, _ dflt: Int) -> Int {
    guard let s = ProcessInfo.processInfo.environment[name], let v = Int(s) else { return dflt }
    return v
}

/// Longest text (characters) to hand F5 in a single generation. F5 stays
/// faithful up to roughly this length / ~16s but degrades past it (garbled
/// words, repetition, reference-text leak) on a single long pass. This 200 is
/// our own empirically-tuned ceiling, not an upstream F5 spec. Inputs longer
/// than this are chunked; see `chunkTextForF5`. Override with
/// `F5_MAX_CHUNK_CHARS` (<=0 disables chunking → one pass over the whole input).
private var F5MaxChunkChars: Int { envInt("F5_MAX_CHUNK_CHARS", 200) }

/// Silence inserted between synthesized chunks (a natural inter-chunk pause and
/// a clean join — see handleF5Clone). Override with `F5_CHUNK_GAP_SECONDS`.
private var F5ChunkGapSeconds: Double { envDouble("F5_CHUNK_GAP_SECONDS", 0.06) }

/// Clamp band (chars/sec) for F5's explicit duration estimate. Override with
/// `F5_CPS_MIN` / `F5_CPS_MAX`; set `F5_DURATION_NATIVE=1` to ignore the clamp
/// and use F5's own duration_v2 predictor instead.
private var F5CpsMin: Double { envDouble("F5_CPS_MIN", 13.0) }
private var F5CpsMax: Double { envDouble("F5_CPS_MAX", 18.0) }
private var F5DurationNative: Bool { (ProcessInfo.processInfo.environment["F5_DURATION_NATIVE"] ?? "") == "1" }

/// Split input into chunks small enough for F5 to render cleanly, preferring
/// sentence boundaries. Sentences are packed greedily up to `maxChars`; a single
/// sentence longer than `maxChars` is split further at word boundaries.
func chunkTextForF5(_ input: String, maxChars: Int) -> [String] {
    var chunks: [String] = []
    var current = ""
    func flush() {
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { chunks.append(t) }
        current = ""
    }
    for sentence in splitIntoSentences(input) {
        if sentence.count > maxChars {
            flush()
            chunks.append(contentsOf: splitLongRun(sentence, maxChars: maxChars))
        } else if current.isEmpty {
            current = sentence
        } else if current.count + 1 + sentence.count <= maxChars {
            current += " " + sentence
        } else {
            flush()
            current = sentence
        }
    }
    flush()
    return chunks.isEmpty ? [input] : chunks
}

/// Split a single over-long sentence into <=maxChars pieces. Prefer breaking at
/// clause punctuation (comma, semicolon, colon, em/en dash) so the cuts land
/// where a speaker would naturally pause; only a clause that is *still* longer
/// than maxChars falls back to word-boundary splitting. Keeping the punctuation
/// with its clause preserves the prosodic cue F5 renders the pause from.
private func splitLongRun(_ sentence: String, maxChars: Int) -> [String] {
    let clauses = splitIntoClauses(sentence)
    var out: [String] = []
    var current = ""
    func flush() {
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { out.append(t) }
        current = ""
    }
    for clause in clauses {
        if clause.count > maxChars {
            // Clause itself too long even after punctuation breaks — fall back
            // to packing its words.
            flush()
            out.append(contentsOf: splitAtWords(clause, maxChars: maxChars))
        } else if current.isEmpty {
            current = clause
        } else if current.count + 1 + clause.count <= maxChars {
            current += " " + clause
        } else {
            flush()
            current = clause
        }
    }
    flush()
    return out
}

/// Break a sentence at clause-level punctuation (, ; : — –), keeping the
/// delimiter attached to the clause it closes. Returns the original sentence as
/// a single element when it has no such punctuation.
private func splitIntoClauses(_ sentence: String) -> [String] {
    let breakers: Set<Character> = [",", ";", ":", "\u{2014}", "\u{2013}"]
    var out: [String] = []
    var current = ""
    for ch in sentence {
        current.append(ch)
        if breakers.contains(ch) {
            let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { out.append(t) }
            current = ""
        }
    }
    let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
    if !tail.isEmpty { out.append(tail) }
    return out.isEmpty ? [sentence] : out
}

/// Last-resort split of an over-long run into <=maxChars pieces at word
/// boundaries (a single word longer than maxChars is left intact).
private func splitAtWords(_ run: String, maxChars: Int) -> [String] {
    var out: [String] = []
    var current = ""
    for word in run.split(separator: " ") {
        if current.isEmpty {
            current = String(word)
        } else if current.count + 1 + word.count <= maxChars {
            current += " " + word
        } else {
            out.append(current)
            current = String(word)
        }
    }
    if !current.isEmpty { out.append(current) }
    return out
}

/// Trim leading/trailing near-silence from a chunk so chunks butt-join at the
/// speech, not on F5's variable edge silence/breath. Threshold is absolute
/// amplitude on the normalized [-1, 1] samples.
func trimEdgeSilence(_ samples: [Float], threshold: Float = 0.015) -> [Float] {
    guard let first = samples.firstIndex(where: { abs($0) > threshold }),
          let last = samples.lastIndex(where: { abs($0) > threshold }) else { return [] }
    return Array(samples[first...last])
}

/// F5 voice clone. F5 renders continuous multi-sentence speech cleanly only up
/// to ~140 chars; beyond that a single pass garbles/repeats/leaks the reference
/// text. So we chunk longer inputs (at sentence boundaries) and synthesize each
/// chunk in one pass, joining them with a short silence after edge-trimming —
/// which avoids BOTH the long-pass degradation and the raw per-sentence
/// concatenation clicks ("clatter") that splitting every sentence produced.
/// Each chunk streams out as it finishes. F5 has no style/`instruct` parameter,
/// so `instructions` is not threaded here.
func handleF5Clone(
    input: String,
    cloneRef: String,
    cloneRefText: String?,
    responseFormat: String,
    speed: Double = 1.0
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
    // F5_MAX_CHUNK_CHARS<=0 disables chunking: hand the whole input to F5 in one
    // pass (experiment knob — see the cps/native knobs below).
    let maxChunk = F5MaxChunkChars
    let chunks = maxChunk > 0 ? chunkTextForF5(input, maxChars: maxChunk) : [input]
    let gap = [Float](repeating: 0, count: Int(F5ChunkGapSeconds * Double(sampleRate)))
    let cpsMin = F5CpsMin
    let cpsMax = F5CpsMax
    let durationNative = F5DurationNative

    return Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init { writer in
            do {
                if format == "wav" {
                    try await writer.write(
                        ByteBuffer(bytes: streamingWAVHeader(sampleRate: sampleRate)))
                }
                var wroteAudio = false
                for chunk in chunks {
                    let out = try await model.generate(
                        text: chunk,
                        referenceAudio: refAudio,
                        referenceAudioText: refText,
                        cpsMin: cpsMin,
                        cpsMax: cpsMax,
                        speed: speed,
                        useNativeDuration: durationNative)
                    let samples = trimEdgeSilence(out.asArray(Float.self))
                    guard !samples.isEmpty else { continue }
                    // Clean silence between chunks (a natural pause) avoids the
                    // waveform-discontinuity click of butt-joining F5 outputs.
                    if wroteAudio {
                        try await writer.write(ByteBuffer(bytes: float32ToPCM16LE(gap)))
                    }
                    try await writer.write(ByteBuffer(bytes: float32ToPCM16LE(samples)))
                    wroteAudio = true
                }
                try await writer.finish(nil)
            } catch {
                try? await writer.finish(nil)
                throw error
            }
        })
}
