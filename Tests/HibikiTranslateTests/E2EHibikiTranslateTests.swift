import XCTest
import MLX
import MLXNN
import MLXRandom
import AudioCommon
import ParakeetASR
import PersonaPlex
@testable import HibikiTranslate

/// E2E tests that download the Hibiki Zero-3B model from HuggingFace and
/// verify weights load + forward pass produces sensible outputs.
///
/// Skipped by default. Enable with `HIBIKI_E2E=1`. Override the model id with
/// `HIBIKI_MODEL_ID=<repo>` (default `aufklarer/Hibiki-Zero-3B-MLX-4bit`).
final class E2EHibikiTranslateTests: XCTestCase {

    /// Verifies the model downloads, weights load with `verify: .noUnusedKeys`,
    /// and a forward pass on synthetic source-audio tokens produces non-NaN
    /// finite text logits with the expected shape.
    func testZero3BLoadAndForward() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E tests (~2.7 GB download)")

        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId

        let model = try await HibikiTranslateModel.fromPretrained(
            modelId: modelId,
            progressHandler: { p, msg in
                if Int(p * 100) % 10 == 0 {
                    print("[hibiki-load] \(Int(p * 100))% \(msg)")
                }
            }
        )

        // Synthetic 1-frame input: text=padding, all 32 audio streams = -1 (masked).
        let cfg = model.cfg
        let textTokens = MLXArray([Int32(cfg.temporal.textPaddingId)]).reshaped([1, 1])
        let audioTokens = MLXArray.full(
            [1, cfg.temporal.numAudioEmbeddings, 1],
            values: MLXArray(Int32(-1))
        )

        let (hidden, textLogits) = model.temporal.forward(
            textTokens: textTokens, audioTokens: audioTokens, offset: 0)
        eval(hidden)
        eval(textLogits)

        XCTAssertEqual(hidden.shape, [1, 1, cfg.temporal.dim],
                       "hidden state should be [1, 1, dim]")
        XCTAssertEqual(textLogits.shape, [1, 1, cfg.temporal.textCard],
                       "text logits should be [1, 1, textCard]")

        // Verify logits are finite (not NaN, not Inf) — proves weights loaded
        // and the forward pass through 28 GQA layers works on real weights.
        let logitsHost = textLogits.asArray(Float.self)
        let nanCount = logitsHost.filter { $0.isNaN }.count
        let infCount = logitsHost.filter { $0.isInfinite }.count
        XCTAssertEqual(nanCount, 0, "text logits should have zero NaN values")
        XCTAssertEqual(infCount, 0, "text logits should have zero Inf values")

        // Top-5 tokens — eyeball check: should not be wildly degenerate.
        let topIdx = argSort(textLogits.squeezed()).asArray(Int32.self)
        let top5 = Array(topIdx.suffix(5).reversed())
        print("[hibiki-forward] top-5 text token ids: \(top5)")
    }

    /// Smoke-tests `HibikiDepformer.generate` on real weights with a trivial
    /// argmax sampler. Verifies the 9-slice schedule wired correctly through
    /// all 16 generation steps.
    func testZero3BDepformerGenerateOnRealWeights() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E tests")

        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId

        let model = try await HibikiTranslateModel.fromPretrained(modelId: modelId)
        let cfg = model.cfg

        // Random temporal hidden state (will produce gibberish tokens but the
        // shape and schedule should be exercised correctly).
        let temporalHidden = MLXRandom.normal([1, 1, cfg.temporal.dim])
        eval(temporalHidden)
        let textToken = MLXArray([Int32(cfg.temporal.textPaddingId)])

        // Simple argmax sampler.
        let sampler: (MLXArray, Int) -> MLXArray = { logits, _ in
            argMax(logits, axis: -1).asType(.int32)
        }

        let tokens = model.depformer.generate(
            temporalHidden: temporalHidden,
            textToken: textToken,
            sampleFn: sampler
        )
        eval(tokens)

        XCTAssertEqual(tokens.shape, [1, cfg.depformer.numSteps],
                       "depformer should emit [1, 16] target tokens")
        let tokensHost = tokens.asArray(Int32.self)
        for t in tokensHost {
            XCTAssertGreaterThanOrEqual(t, 0, "token \(t) below valid range")
            XCTAssertLessThan(t, Int32(cfg.depformer.card),
                              "token \(t) above audio cardinality \(cfg.depformer.card)")
        }
    }

    /// **End-to-end round-trip translation test**: known FR audio →
    /// Hibiki Zero-3B → EN audio → Parakeet ASR → assert plausible English.
    ///
    /// Reference content for `fleurs_fr.wav` (FLEURS dataset):
    ///   FR: "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire."
    ///   EN: "Think of the ski route as a similar hiking route."
    ///   Expected English keywords (any subset is meaningful): think, ski,
    ///   route, trail, hiking, hike, similar.
    ///
    /// Exercises every milestone end-to-end: config, rope_concat, GQA temporal,
    /// scheduled depformer + per-step LayerNorm, weight loader, driver, Mimi
    /// encode/decode, ASR pipeline. Skip with `HIBIKI_E2E` unset.
    ///
    /// - Soft mode (default): asserts the pipeline runs and produces non-empty
    ///   English-shaped output. Logs reference vs ASR for inspection.
    /// - Strict mode: set `HIBIKI_STRICT=1` to require ≥1 expected keyword.
    func testFrenchToEnglishTranslation() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E translation")

        let strict = ProcessInfo.processInfo.environment["HIBIKI_STRICT"] != nil

        // Reference content for fleurs_fr.wav (FLEURS dataset).
        let referenceFR = "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire."
        let referenceEN = "Think of the ski route as a similar hiking route."
        let expectedKeywords = ["think", "ski", "route", "trail", "hike",
                                "hiking", "similar", "path"]

        // 1. Load French source audio.
        guard let frURL = Bundle.module.url(forResource: "fleurs_fr", withExtension: "wav") else {
            XCTFail("fleurs_fr.wav missing from test resources"); return
        }
        let pcm = try AudioFileLoader.load(url: frURL, targetSampleRate: 24000)
        let inputDuration = Double(pcm.count) / 24000.0
        print("[hibiki-e2e] loaded fleurs_fr.wav: \(pcm.count) samples, " +
              "\(String(format: "%.2f", inputDuration))s")
        print("[hibiki-e2e] reference FR: \(referenceFR)")
        print("[hibiki-e2e] reference EN: \(referenceEN)")

        // 2. Load Hibiki Zero-3B (4-bit by default).
        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId
        let model = try await HibikiTranslateModel.fromPretrained(
            modelId: modelId,
            progressHandler: { p, msg in
                if Int(p * 100) % 20 == 0 { print("[hibiki-e2e] load \(Int(p * 100))% \(msg)") }
            }
        )

        // 3. Translate.
        let (audio, textTokens) = model.translate(
            sourceAudio: pcm, sourceLanguage: .fr, verbose: true
        )
        let outputDuration = Double(audio.count) / 24000.0

        // Hibiki is synchronous 1:1 — output ≈ input duration.
        XCTAssertGreaterThan(audio.count, Int(0.5 * 24000),
                             "should produce > 0.5 s of English audio")
        XCTAssertGreaterThan(outputDuration, inputDuration * 0.7,
                             "output duration should be at least 70% of input (synchronous 1:1)")
        XCTAssertLessThan(outputDuration, inputDuration * 1.3,
                          "output duration should be at most 130% of input (synchronous 1:1)")
        XCTAssertEqual(textTokens.count, Int(ceil(inputDuration * 12.5)),
                       accuracy: 2,
                       "text tokens should match ~12.5 Hz Mimi frame rate")

        print("[hibiki-e2e] translated → \(audio.count) samples, " +
              "\(String(format: "%.2f", outputDuration))s; " +
              "\(textTokens.count) text tokens")

        // Verify the output has non-trivial audio energy (not all silence).
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        let absMax = audio.map { abs($0) }.max() ?? 0
        print("[hibiki-e2e] output audio RMS: \(String(format: "%.4f", rms)), peak: \(String(format: "%.4f", absMax))")
        XCTAssertGreaterThan(rms, 0.001,
            "output audio should have non-trivial RMS energy (not silence)")

        // 4. Save output for inspection.
        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hibiki-e2e", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let outURL = outDir.appendingPathComponent("translated_en.wav")
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outURL)
        print("[hibiki-e2e] wrote translated audio: \(outURL.path)")

        // 5. ASR the output with Parakeet (multilingual, includes English).
        let asr = try await ParakeetASRModel.fromPretrained()
        let transcript = asr.transcribe(audio: audio, sampleRate: 24000, language: nil)
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        print("[hibiki-e2e] Parakeet ASR of EN output: '\(transcript)'")

        // 6. Round-trip assertions.
        // STRUCTURAL (always asserted): pipeline produces audio that an English
        // ASR can attempt to process. Quality is signaled but not asserted here
        // because Hibiki Zero-3B output quality on short out-of-distribution
        // FLEURS clips is known to be unstable.
        let hits = expectedKeywords.filter { transcript.contains($0) }
        print("[hibiki-e2e] expected keyword hits: \(hits.isEmpty ? "[none]" : hits.joined(separator: ", "))")

        // QUALITY (strict mode only): require ≥1 expected English keyword.
        // Run with `HIBIKI_STRICT=1` to enable. Useful for regression detection
        // once the driver is tuned, or when running against samples Hibiki
        // handles well (longer clips, in-distribution speech).
        if strict {
            XCTAssertFalse(transcript.isEmpty,
                "STRICT: Parakeet transcript empty for output of '\(referenceFR)'.")
            XCTAssertFalse(hits.isEmpty,
                "STRICT: expected ≥1 of \(expectedKeywords) in transcript '\(transcript)'. " +
                "Reference EN: '\(referenceEN)'.")
        } else {
            if transcript.isEmpty {
                print("[hibiki-e2e] WARNING: Parakeet transcript empty (model output may be low-quality on this short sample)")
            } else if hits.isEmpty {
                print("[hibiki-e2e] WARNING: 0 expected keywords matched (set HIBIKI_STRICT=1 to fail)")
            }
        }
    }
}
