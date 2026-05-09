import Foundation
import MLX
import MLXNN
import PersonaPlex   // sampleTextWithPenalty, sampleTopKWithPenalty
import AudioCommon

// MARK: - Streaming config

public struct HibikiStreamingConfig: Sendable {
    /// Frames accumulated before the first chunk is emitted.
    public var firstChunkFrames: Int
    /// Frames per subsequent chunk.
    public var chunkFrames: Int

    public init(firstChunkFrames: Int = 25, chunkFrames: Int = 25) {
        self.firstChunkFrames = firstChunkFrames
        self.chunkFrames = chunkFrames
    }

    public static let `default` = HibikiStreamingConfig()
}

// MARK: - Driver

public extension HibikiTranslateModel {

    /// Translate source-language speech to English speech (offline).
    ///
    /// Hibiki is **synchronous 1:1**: each Mimi frame of input (80 ms) produces
    /// exactly one Mimi frame of output. So `output_duration ≈ input_duration`.
    ///
    /// - Parameters:
    ///   - sourceAudio: PCM samples at 24 kHz, mono.
    ///   - sourceLanguage: hint only — Hibiki Zero auto-detects.
    ///   - verbose: print per-phase timings.
    /// - Returns: tuple of (English audio samples at 24 kHz, generated text token IDs).
    func translate(
        sourceAudio: [Float],
        sourceLanguage: HibikiSourceLanguage = .fr,
        verbose: Bool = false
    ) -> (audio: [Float], textTokens: [Int32]) {
        let totalStart = CFAbsoluteTimeGetCurrent()
        let nQ = cfg.temporal.nQ                                     // 16
        let numStreams = cfg.numStreams                              // 33
        let delays = cfg.delays                                      // 33 entries
        let maxDelay = cfg.maxDelay                                  // 2

        // 1. Encode source audio with Mimi.
        let encStart = CFAbsoluteTimeGetCurrent()
        let audioMLX = MLXArray(sourceAudio).reshaped([1, 1, sourceAudio.count])
        let sourceCodes = mimi.encode(audioMLX)   // [1, 16, T_src]
        eval(sourceCodes)
        let tSrc = sourceCodes.shape[2]
        guard tSrc > 0 else { return ([], []) }
        if verbose {
            print("  Mimi encode: \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - encStart))s, frames: \(tSrc)")
        }

        // 2. Initialize 33-stream token cache. Length covers all source frames
        // plus the maximum delay so delay-2 streams can be written ahead.
        let totalLen = tSrc + maxDelay + 2
        var tokenCache = [[Int32]](
            repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // Pre-populate text stream with padding tokens.
        for t in 0..<tSrc {
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
        }

        // Pre-populate source codebook streams (1..16) with Mimi-encoded codes.
        // Each stream s gets shifted by delays[s].
        let sourceCodesHost = sourceCodes.reshaped([nQ, tSrc]).asArray(Int32.self)
        for cb in 0..<nQ {
            let streamIdx = 1 + cb
            let delay = delays[streamIdx]
            for t in 0..<tSrc {
                let writePos = t + delay
                if writePos < totalLen {
                    tokenCache[streamIdx][writePos] = sourceCodesHost[cb * tSrc + t]
                }
            }
        }

        // 3. Generation loop.
        var allTextTokens: [Int32] = []
        var targetTokens: [[Int32]] = (0..<nQ).map { _ in [] }
        var perCodebookHistory: [[Int32]] = (0..<cfg.depformer.numSteps).map { _ in [] }

        let genStart = CFAbsoluteTimeGetCurrent()
        temporal.resetCache()

        // Hibiki runs exactly tSrc steps (synchronous 1:1).
        for step in 0..<tSrc {
            if Task.isCancelled { break }

            // Read previous-step input for all 33 streams. Source streams
            // 1..16 are pre-populated; target streams 17..32 are filled from
            // the previous depformer output.
            let readIdx = step - 1
            let textTok = readIdx >= 0 ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])

            var audioStreamTokens: [Int32] = []
            for stream in 1..<numStreams {
                let tok = readIdx >= 0 ? tokenCache[stream][readIdx] : Int32(-1)
                audioStreamTokens.append(tok)
            }
            let audioTokens = MLXArray(audioStreamTokens)
                .reshaped([1, numStreams - 1, 1])

            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr,
                audioTokens: audioTokens,
                offset: step
            )

            // Sample text token (with optional repetition penalty).
            let textHistory = Array(allTextTokens.suffix(cfg.sampling.repetitionWindow))
            let textToken = sampleTextWithPenalty(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp,
                topK: cfg.sampling.textTopK,
                pastTokens: textHistory,
                penalty: cfg.sampling.textRepetitionPenalty
            )
            eval(textToken)
            let textVal = textToken[0].item(Int32.self)
            tokenCache[0][step] = textVal
            allTextTokens.append(textVal)

            // Generate the 16 target codebooks via depformer.
            let targetCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken
            ) { logits, cbIdx in
                let history = Array(perCodebookHistory[cbIdx].suffix(cfg.sampling.repetitionWindow))
                return sampleTopKWithPenalty(
                    logits: logits,
                    temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK,
                    pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty
                )
            }

            // Write target codebooks into streams 17..32 (with delay shifts).
            let codes = targetCodes[0]   // [16]
            for cb in 0..<nQ {
                let tok = codes[cb].item(Int32.self)
                targetTokens[cb].append(tok)
                perCodebookHistory[cb].append(tok)
                let streamIdx = 1 + nQ + cb
                let delay = delays[streamIdx]
                let writePos = step + delay
                if writePos < totalLen {
                    tokenCache[streamIdx][writePos] = tok
                }
            }
        }
        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let msPerStep = tSrc > 0 ? genTime / Double(tSrc) * 1000 : 0
            print("  Generation: \(String(format: "%.2f", genTime))s, " +
                  "\(String(format: "%.1f", msPerStep))ms/step (\(tSrc) steps)")
        }

        // 4. Decode target codebooks via Mimi → English audio.
        let decStart = CFAbsoluteTimeGetCurrent()
        let numFrames = targetTokens[0].count
        guard numFrames > 0 else { return ([], allTextTokens) }

        var flat: [Int32] = []
        flat.reserveCapacity(nQ * numFrames)
        for cb in 0..<nQ { flat.append(contentsOf: targetTokens[cb]) }
        let codesArr = MLXArray(flat).reshaped([1, nQ, numFrames])
        let decoded = mimi.decode(codesArr)   // [1, 1, samples]
        eval(decoded)

        let numSamples = decoded.shape[2]
        let samples = decoded.reshaped([numSamples]).asArray(Float.self)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
            let total = CFAbsoluteTimeGetCurrent() - totalStart
            let dur = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", total))s, " +
                  "audio: \(String(format: "%.2f", dur))s, " +
                  "RTF: \(String(format: "%.2f", total / max(dur, 0.001)))")
        }
        _ = sourceLanguage   // unused — model auto-detects, kept for API completeness
        return (samples, allTextTokens)
    }

    /// Streaming counterpart that emits `AudioChunk`s as Mimi finishes decoding
    /// each accumulated batch of target frames.
    func translateStream(
        sourceAudio: [Float],
        sourceLanguage: HibikiSourceLanguage = .fr,
        streaming: HibikiStreamingConfig = .default,
        verbose: Bool = false
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let cfg = self.cfg
        let model = self
        return AsyncThrowingStream<AudioChunk, Error> { continuation in
            Task {
                do {
                    // For v1, run offline translate then emit a single chunk
                    // covering the full output. True streaming via decodeStep
                    // can be added in a follow-up. Note that Hibiki's
                    // synchronous 1:1 generation means we know the full output
                    // duration up front.
                    let (audio, text) = model.translate(
                        sourceAudio: sourceAudio,
                        sourceLanguage: sourceLanguage,
                        verbose: verbose
                    )
                    let chunk = AudioChunk(
                        samples: audio,
                        sampleRate: cfg.sampleRate,
                        frameIndex: 0,
                        isFinal: true,
                        textTokens: text
                    )
                    continuation.yield(chunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
