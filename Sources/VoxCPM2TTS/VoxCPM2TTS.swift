import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN
import Tokenizers

public final class ScalarQuantizationLayer: Module {
    @ModuleInfo(key: "in_proj") public var in_proj: Linear
    @ModuleInfo(key: "out_proj") public var out_proj: Linear
    public let scale: Int

    public init(inDim: Int, outDim: Int, latentDim: Int = 64, scale: Int = 9) {
        self.scale = scale
        self._in_proj = ModuleInfo(wrappedValue: Linear(inDim, latentDim), key: "in_proj")
        self._out_proj = ModuleInfo(wrappedValue: Linear(latentDim, outDim), key: "out_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = in_proj(x)
        let quantized = round(tanh(h) * MLXArray(Float(scale))) / MLXArray(Float(scale))
        return out_proj(quantized)
    }
}

public final class VoxCPM2TTSModel: Module {
    public let args: ModelArgs
    public let outputSampleRate: Int

    @ModuleInfo public var base_lm: MiniCPMModel
    @ModuleInfo public var residual_lm: MiniCPMModel
    @ModuleInfo public var feat_encoder: VoxCPMLocEnc
    @ModuleInfo public var feat_decoder: UnifiedCFM
    @ModuleInfo public var fsq_layer: ScalarQuantizationLayer
    @ModuleInfo public var enc_to_lm_proj: Linear
    @ModuleInfo public var lm_to_dit_proj: Linear
    @ModuleInfo public var res_to_dit_proj: Linear
    @ModuleInfo public var fusion_concat_proj: Linear
    @ModuleInfo public var stop_proj: Linear
    @ModuleInfo public var stop_head: Linear
    @ModuleInfo public var audio_vae: AudioVAE

    private var tokenizer: Tokenizer?
    private var _isLoaded: Bool = true

    public init(args: ModelArgs) {
        self.args = args
        self.outputSampleRate = args.audioVAEConfig.outSampleRate

        let lmConfig = args.lmConfig
        self._base_lm = ModuleInfo(wrappedValue: MiniCPMModel(lmConfig))

        var residualConfig = lmConfig
        residualConfig.numHiddenLayers = args.residualLMNumLayers
        residualConfig.vocabSize = 0
        residualConfig.noRope = args.residualLMNoRope
        self._residual_lm = ModuleInfo(wrappedValue: MiniCPMModel(residualConfig))

        var encoderConfig = lmConfig
        encoderConfig.hiddenSize = args.encoderConfig.hiddenDim
        encoderConfig.intermediateSize = args.encoderConfig.ffnDim
        encoderConfig.numAttentionHeads = args.encoderConfig.numHeads
        encoderConfig.numHiddenLayers = args.encoderConfig.numLayers
        encoderConfig.kvChannels = args.encoderConfig.kvChannels
        encoderConfig.vocabSize = 0
        self._feat_encoder = ModuleInfo(wrappedValue: VoxCPMLocEnc(config: encoderConfig, inputDim: args.featDim))

        var ditConfig = lmConfig
        ditConfig.hiddenSize = args.ditConfig.hiddenDim
        ditConfig.intermediateSize = args.ditConfig.ffnDim
        ditConfig.numAttentionHeads = args.ditConfig.numHeads
        ditConfig.numHiddenLayers = args.ditConfig.numLayers
        ditConfig.kvChannels = args.ditConfig.kvChannels
        ditConfig.vocabSize = 0
        let estimator = VoxCPMLocDiTV2(config: ditConfig, inChannels: args.featDim)
        self._feat_decoder = ModuleInfo(wrappedValue: UnifiedCFM(
            inChannels: args.featDim,
            cfmParams: args.ditConfig.cfmConfig,
            estimator: estimator,
            meanMode: args.ditConfig.ditMeanMode
        ))

        self._fsq_layer = ModuleInfo(wrappedValue: ScalarQuantizationLayer(
            inDim: args.lmConfig.hiddenSize,
            outDim: args.lmConfig.hiddenSize,
            latentDim: args.scalarQuantizationLatentDim,
            scale: args.scalarQuantizationScale
        ))

        self._enc_to_lm_proj = ModuleInfo(wrappedValue: Linear(args.encoderConfig.hiddenDim, args.lmConfig.hiddenSize))
        self._lm_to_dit_proj = ModuleInfo(wrappedValue: Linear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim))
        self._res_to_dit_proj = ModuleInfo(wrappedValue: Linear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim))
        self._fusion_concat_proj = ModuleInfo(wrappedValue: Linear(args.lmConfig.hiddenSize * 2, args.lmConfig.hiddenSize))
        self._stop_proj = ModuleInfo(wrappedValue: Linear(args.lmConfig.hiddenSize, args.lmConfig.hiddenSize))
        self._stop_head = ModuleInfo(wrappedValue: Linear(args.lmConfig.hiddenSize, 2, bias: false))
        self._audio_vae = ModuleInfo(wrappedValue: AudioVAE(args.audioVAEConfig))

        super.init()
    }

    // MARK: - Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/VoxCPM2-MLX-bf16",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VoxCPM2TTSModel {
        let modelCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        if !HuggingFaceDownloader.weightsExist(in: modelCacheDir) || !FileManager.default.fileExists(atPath: modelCacheDir.appendingPathComponent("config.json").path) {
            progressHandler?(0.0, "Downloading \(modelId)...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: modelCacheDir,
                additionalFiles: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "generation_config.json"
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.8, "Downloading model...")
            }
        }

        progressHandler?(0.82, "Loading config...")
        let args = try ModelArgs.load(from: modelCacheDir)
        let model = VoxCPM2TTSModel(args: args)

        progressHandler?(0.88, "Loading tokenizer...")
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelCacheDir)
        model.setTokenizer(tokenizer)

        progressHandler?(0.92, "Loading weights...")
        try model.loadWithDiagnostics("loadWeights(from:)") {
            try model.loadWeights(from: modelCacheDir)
        }

        progressHandler?(0.98, "Evaluating parameters...")
        // NOTE: MLX can trip over some parameter views during eager evaluation
        // even when the model loads cleanly. We defer evaluation until first use.

        progressHandler?(1.0, "Ready")
        return model
    }

    private func swapQuantizedLinears(from weights: [String: MLXArray]) throws {
        let quantizedPaths = Set(weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        guard !quantizedPaths.isEmpty else { return }

        let bits: Int
        if let cfg = args.quantization, cfg.isQuantized {
            bits = cfg.bits
        } else {
            bits = inferQuantizationBits(from: weights) ?? 4
        }
        let groupSize = args.quantization?.groupSize ?? 64
        guard bits == 4 || bits == 8 else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported quantization bits: \(bits) (expected 4 or 8)"
            ])
        }

        // Walk every leaf module; convert each Linear whose path appears in the
        // safetensors as a quantized weight. `quantize(model:filter:)` from MLXNN
        // does the Linear → QuantizedLinear swap in place via update(modules:).
        // The placeholder quantized parameters are overwritten by the subsequent
        // update(parameters:) calls in loadWeights().
        quantize(
            model: self,
            groupSize: groupSize,
            bits: bits,
            filter: { path, _ in quantizedPaths.contains(path) }
        )
    }

    private func inferQuantizationBits(from weights: [String: MLXArray]) -> Int? {
        for key in weights.keys where key.hasSuffix(".scales") {
            let prefix = String(key.dropLast(".scales".count))
            guard let weight = weights["\(prefix).weight"],
                  let scales = weights["\(prefix).scales"],
                  weight.ndim == 2, scales.ndim == 2 else { continue }
            let packedCols = weight.dim(1)
            let numGroups = scales.dim(1)
            guard numGroups > 0 else { continue }
            let ratio = packedCols / numGroups
            if ratio == 8 { return 4 }
            if ratio == 16 { return 8 }
        }
        return nil
    }

    private func loadWeights(from directory: URL) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        let sanitized = audio_vae.sanitize(allWeights)
        try swapQuantizedLinears(from: sanitized)
        try loadWithDiagnostics("base_lm") {
            try loadWeights(into: base_lm, prefix: "base_lm", from: sanitized)
        }
        try loadWithDiagnostics("residual_lm") {
            try loadWeights(into: residual_lm, prefix: "residual_lm", from: sanitized)
        }
        if let specialToken = sanitized["feat_encoder.special_token"] {
            try loadWithDiagnostics("feat_encoder.special_token") {
                feat_encoder.loadSpecialToken(specialToken)
            }
        }
        try loadWithDiagnostics("feat_encoder.in_proj") {
            try loadLinearWeights(to: feat_encoder.inProj, prefix: "feat_encoder.in_proj", from: sanitized)
        }
        try loadWithDiagnostics("feat_encoder.encoder") {
            try loadWeights(into: feat_encoder.encoder, prefix: "feat_encoder.encoder", from: sanitized)
        }
        try loadWithDiagnostics("feat_decoder") {
            try loadWeights(into: feat_decoder, prefix: "feat_decoder", from: sanitized)
        }
        try loadWithDiagnostics("fsq_layer") {
            try loadWeights(into: fsq_layer, prefix: "fsq_layer", from: sanitized)
        }
        try loadWithDiagnostics("enc_to_lm_proj") {
            try loadLinearWeights(to: enc_to_lm_proj, prefix: "enc_to_lm_proj", from: sanitized)
        }
        try loadWithDiagnostics("lm_to_dit_proj") {
            try loadLinearWeights(to: lm_to_dit_proj, prefix: "lm_to_dit_proj", from: sanitized)
        }
        try loadWithDiagnostics("res_to_dit_proj") {
            try loadLinearWeights(to: res_to_dit_proj, prefix: "res_to_dit_proj", from: sanitized)
        }
        try loadWithDiagnostics("fusion_concat_proj") {
            try loadLinearWeights(to: fusion_concat_proj, prefix: "fusion_concat_proj", from: sanitized)
        }
        try loadWithDiagnostics("stop_proj") {
            try loadLinearWeights(to: stop_proj, prefix: "stop_proj", from: sanitized)
        }
        try loadWithDiagnostics("stop_head") {
            try loadLinearWeights(to: stop_head, prefix: "stop_head", from: sanitized)
        }
        try loadWithDiagnostics("audio_vae.encoder.conv_in") {
            if let weight = sanitized["audio_vae.encoder.conv_in.weight"] {
                try ensureShape(weight, matches: audio_vae.encoder.conv_in.weight.shape, label: "audio_vae.encoder.conv_in.weight")
                audio_vae.encoder.conv_in.weight = weight
            }
            if let bias = sanitized["audio_vae.encoder.conv_in.bias"] {
                if let currentBias = audio_vae.encoder.conv_in.bias {
                    try ensureShape(bias, matches: currentBias.shape, label: "audio_vae.encoder.conv_in.bias")
                }
                audio_vae.encoder.conv_in.bias = bias
            }
        }
        try loadWithDiagnostics("audio_vae.decoder.conv_in") {
            try loadDecoderConvStack(audio_vae.decoder.conv_in, prefix: "audio_vae.decoder.conv_in", from: sanitized)
        }
        for (index, block) in audio_vae.decoder.blocks.layers.enumerated() {
            try loadWithDiagnostics("audio_vae.decoder.blocks.layers.\(index)") {
                try loadDecoderBlock(block, prefix: "audio_vae.decoder.blocks.layers.\(index)", from: sanitized)
            }
        }
        try loadWithDiagnostics("audio_vae.decoder.snake_out/conv_out") {
            if let snakeOut = sanitized["audio_vae.decoder.snake_out.alpha"] {
                try ensureShape(
                    snakeOut,
                    matches: audio_vae.decoder.snake_out.alpha.shape,
                    label: "audio_vae.decoder.snake_out.alpha"
                )
                audio_vae.decoder.snake_out.loadAlpha(snakeOut)
            }
            if let convOutWeight = sanitized["audio_vae.decoder.conv_out.weight"] {
                try ensureShape(
                    convOutWeight,
                    matches: audio_vae.decoder.conv_out.weight.shape,
                    label: "audio_vae.decoder.conv_out.weight"
                )
                audio_vae.decoder.conv_out.weight = convOutWeight
            }
            if let convOutBias = sanitized["audio_vae.decoder.conv_out.bias"] {
                if let currentBias = audio_vae.decoder.conv_out.bias {
                    try ensureShape(
                        convOutBias,
                        matches: currentBias.shape,
                        label: "audio_vae.decoder.conv_out.bias"
                    )
                }
                audio_vae.decoder.conv_out.bias = convOutBias
            }
        }
    }

    private func loadWithDiagnostics<T>(_ label: String, _ body: () throws -> T) throws -> T {
        do {
            return try withErrorHandler({ message in
                if let data = "[VoxCPM2] MLX error in \(label): \(message)\n".data(using: .utf8) {
                    FileHandle.standardOutput.write(data)
                }
            }) {
                try withError { error in
                    let value = try body()
                    try error.check()
                    return value
                }
            }
        } catch {
            if let data = "[VoxCPM2] Swift error in \(label): \(error)\n".data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
            throw error
        }
    }

    private func loadLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }
        guard !params.isEmpty else { return }
        try linear.update(parameters: ModuleParameters(values: params), verify: .shapeMismatch)
    }

    private func loadWeights<T: Module>(
        into module: T,
        prefix: String,
        from weights: [String: MLXArray],
        stripPrefix: String? = nil
    ) throws {
        let prefixWithDot = prefix + "."
        let stripPrefix = stripPrefix ?? prefixWithDot
        let filtered = weights.reduce(into: [String: MLXArray]()) { result, entry in
            if entry.key == prefix {
                result[""] = entry.value
            } else if entry.key.hasPrefix(prefixWithDot) {
                result[String(entry.key.dropFirst(stripPrefix.count))] = entry.value
            }
        }

        guard !filtered.isEmpty else {
            return
        }

        if let snake = module as? Snake1d {
            if let alpha = filtered["alpha"] ?? filtered[""] {
                try loadWithDiagnostics(prefix) {
                    snake.loadAlpha(alpha)
                }
                return
            }
        }

        let params = ModuleParameters.unflattened(filtered)
        _ = try loadWithDiagnostics(prefix) {
            try module.update(parameters: params, verify: .shapeMismatch)
        }
    }

    private func loadDecoderConvStack(
        _ stack: ConvStack1d,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        for (index, layer) in stack.layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(index)"
            if let weight = weights["\(layerPrefix).weight"] {
                try ensureShape(weight, matches: layer.weight.shape, label: "\(layerPrefix).weight")
                layer.weight = weight
            }
            if let bias = weights["\(layerPrefix).bias"] {
                if let currentBias = layer.bias {
                    try ensureShape(bias, matches: currentBias.shape, label: "\(layerPrefix).bias")
                }
                layer.bias = bias
            }
        }
    }

    private func loadResidualUnit(
        _ unit: CausalResidualUnit,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let alpha = weights["\(prefix).snake1.alpha"] {
            try ensureShape(alpha, matches: unit.snake1.alpha.shape, label: "\(prefix).snake1.alpha")
            unit.snake1.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv1.weight"] {
            try ensureShape(weight, matches: unit.conv1.weight.shape, label: "\(prefix).conv1.weight")
            unit.conv1.weight = weight
        }
        if let bias = weights["\(prefix).conv1.bias"] {
            if let currentBias = unit.conv1.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv1.bias")
            }
            unit.conv1.bias = bias
        }
        if let alpha = weights["\(prefix).snake2.alpha"] {
            try ensureShape(alpha, matches: unit.snake2.alpha.shape, label: "\(prefix).snake2.alpha")
            unit.snake2.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv2.weight"] {
            try ensureShape(weight, matches: unit.conv2.weight.shape, label: "\(prefix).conv2.weight")
            unit.conv2.weight = weight
        }
        if let bias = weights["\(prefix).conv2.bias"] {
            if let currentBias = unit.conv2.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv2.bias")
            }
            unit.conv2.bias = bias
        }
    }

    private func loadDecoderBlock(
        _ block: CausalDecoderBlock,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let alpha = weights["\(prefix).snake.alpha"] {
            try ensureShape(alpha, matches: block.snake.alpha.shape, label: "\(prefix).snake.alpha")
            block.snake.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv_t.weight"] {
            try ensureShape(weight, matches: block.conv_t.weight.shape, label: "\(prefix).conv_t.weight")
            block.conv_t.weight = weight
        }
        if let bias = weights["\(prefix).conv_t.bias"] {
            if let currentBias = block.conv_t.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv_t.bias")
            }
            block.conv_t.bias = bias
        }
        try loadResidualUnit(block.res1, prefix: "\(prefix).res1", from: weights)
        try loadResidualUnit(block.res2, prefix: "\(prefix).res2", from: weights)
        try loadResidualUnit(block.res3, prefix: "\(prefix).res3", from: weights)
    }

    private func ensureShape(
        _ array: MLXArray,
        matches expectedShape: [Int],
        label: String
    ) throws {
        guard array.shape == expectedShape else {
            throw NSError(
                domain: "VoxCPM2TTSModel",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(label) shape mismatch: expected \(expectedShape), got \(array.shape)"
                ]
            )
        }
    }

    public func setTokenizer(_ tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    // MARK: - Shape Helpers

    private func tokenize(_ text: String) throws -> [Int32] {
        guard let tokenizer else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Tokenizer not loaded"
            ])
        }
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        return ids.map(Int32.init)
    }

    private func encodeAudio(
        _ audio: [Float],
        sampleRate: Int,
        paddingMode: String = "right"
    ) throws -> MLXArray {
        var mono = audio
        if sampleRate != audio_vae.sampleRate {
            mono = AudioFileLoader.resample(audio, from: sampleRate, to: audio_vae.sampleRate)
        }
        guard !mono.isEmpty else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Reference/prompt audio is empty"
            ])
        }

        let patchLen = args.patchSize * audio_vae.chunkSize
        let remainder = mono.count % patchLen
        if remainder != 0 {
            let pad = patchLen - remainder
            if paddingMode == "left" {
                mono = [Float](repeating: 0, count: pad) + mono
            } else {
                mono += [Float](repeating: 0, count: pad)
            }
        }

        let input = MLXArray(mono).reshaped([1, mono.count, 1])
        let feat = audio_vae.encode(input, sampleRate: audio_vae.sampleRate).squeezed(axis: 0)
        let numPatches = feat.dim(0) / args.patchSize
        return feat.reshaped([numPatches, args.patchSize, audio_vae.latentDim])
    }

    private func makeTimeSpan(_ timesteps: Int) -> [Float] {
        guard timesteps > 0 else { return [1.0, 0.0] }
        let swayCoef: Float = 1.0
        return (0...timesteps).map { step in
            let t = Float(step) / Float(timesteps)
            return t + swayCoef * (cos(Float.pi / 2.0 * t) - 1.0 + t)
        }.reversed()
    }

    // MARK: - Generation

    public func generate(text: String, language: String? = nil) async throws -> [Float] {
        try await generateVoxCPM2(
            text: text,
            language: language,
            maxTokens: 2000,
            minTokens: 2,
            refText: nil,
            refAudio: nil,
            promptText: nil,
            promptAudio: nil,
            inferenceTimesteps: 10,
            cfgValue: 2.0,
            streamingPrefixLen: 4,
            warmupPatches: 0,
            instruct: nil
        )
    }

    public func generateVoxCPM2(
        text: String,
        language: String? = nil,
        maxTokens: Int = 2000,
        minTokens: Int = 2,
        refText: String? = nil,
        refAudio: [Float]? = nil,
        promptText: String? = nil,
        promptAudio: [Float]? = nil,
        inferenceTimesteps: Int = 10,
        cfgValue: Float = 2.0,
        streamingPrefixLen: Int = 4,
        warmupPatches: Int = 0,
        instruct: String? = nil
    ) async throws -> [Float] {
        guard tokenizer != nil else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Tokenizer not loaded"
            ])
        }
        _ = language

        var workingText = text
        var effectiveWarmup = warmupPatches
        if let instruct, !instruct.isEmpty {
            workingText = "(\(instruct))\(workingText)"
            effectiveWarmup = min(effectiveWarmup, 1)
        }

        let scaleEmb = args.lmConfig.useMup ? Float(args.lmConfig.scaleEmb) : 1.0
        let latentDim = audio_vae.latentDim
        let hasRef = refAudio != nil
        let hasPrompt = promptAudio != nil && promptText != nil

        let textIds: [Int32]
        var textToken: MLXArray
        var audioFeat: MLXArray
        var textMask: MLXArray
        var audioMask: MLXArray

        if hasRef && hasPrompt {
            let combinedText = (promptText ?? "") + workingText
            textIds = try tokenize(combinedText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let refFeat = try encodeAudio(
                refAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "right"
            )
            let promptFeat = try encodeAudio(
                promptAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "left"
            )
            let promptLen = promptFeat.dim(0)

            let refTokens = MLXArray(
                [Int32(103)]
                + Array(repeating: Int32(0), count: refFeat.dim(0))
                + [Int32(104)]
            )
            let refFeats = concatenated(
                [
                    MLXArray.zeros([1, args.patchSize, latentDim]),
                    refFeat,
                    MLXArray.zeros([1, args.patchSize, latentDim])
                ],
                axis: 0
            )
            let refTMask = MLXArray(
                [Float(1.0)]
                + Array(repeating: Float(0.0), count: refFeat.dim(0))
                + [Float(1.0)]
            )
            let refAMask = MLXArray(
                [Float(0.0)]
                + Array(repeating: Float(1.0), count: refFeat.dim(0))
                + [Float(0.0)]
            )

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let promptPadToken = MLXArray.zeros([promptLen], dtype: .int32)

            let fullText = concatenated([refTokens, textToken.squeezed(axis: 0), promptPadToken], axis: 0)
            let fullAudio = concatenated([refFeats, textPadFeat, promptFeat], axis: 0)
            textMask = concatenated(
                [
                    refTMask,
                    MLXArray.ones([textLength], dtype: .float32),
                    MLXArray.zeros([promptLen], dtype: .float32)
                ],
                axis: 0
            )
            audioMask = concatenated(
                [
                    refAMask,
                    MLXArray.zeros([textLength], dtype: .float32),
                    MLXArray.ones([promptLen], dtype: .float32)
                ],
                axis: 0
            )

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else if hasRef {
            textIds = try tokenize(workingText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let refFeat = try encodeAudio(
                refAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "right"
            )
            let refTokens = MLXArray(
                [Int32(103)]
                + Array(repeating: Int32(0), count: refFeat.dim(0))
                + [Int32(104)]
            )
            let refFeats = concatenated(
                [
                    MLXArray.zeros([1, args.patchSize, latentDim]),
                    refFeat,
                    MLXArray.zeros([1, args.patchSize, latentDim])
                ],
                axis: 0
            )
            let refTMask = MLXArray(
                [Float(1.0)]
                + Array(repeating: Float(0.0), count: refFeat.dim(0))
                + [Float(1.0)]
            )
            let refAMask = MLXArray(
                [Float(0.0)]
                + Array(repeating: Float(1.0), count: refFeat.dim(0))
                + [Float(0.0)]
            )

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let fullText = concatenated([refTokens, textToken.squeezed(axis: 0)], axis: 0)
            let fullAudio = concatenated([refFeats, textPadFeat], axis: 0)
            textMask = concatenated([refTMask, MLXArray.ones([textLength], dtype: .float32)], axis: 0)
            audioMask = concatenated([refAMask, MLXArray.zeros([textLength], dtype: .float32)], axis: 0)

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else if hasPrompt {
            let combinedText = (promptText ?? "") + workingText
            textIds = try tokenize(combinedText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let promptFeat = try encodeAudio(
                promptAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "left"
            )
            let promptLen = promptFeat.dim(0)

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let promptPadToken = MLXArray.zeros([promptLen], dtype: .int32)

            let fullText = concatenated([textToken.squeezed(axis: 0), promptPadToken], axis: 0)
            let fullAudio = concatenated([textPadFeat, promptFeat], axis: 0)
            textMask = concatenated(
                [
                    MLXArray.ones([textLength], dtype: .float32),
                    MLXArray.zeros([promptLen], dtype: .float32)
                ],
                axis: 0
            )
            audioMask = concatenated(
                [
                    MLXArray.zeros([textLength], dtype: .float32),
                    MLXArray.ones([promptLen], dtype: .float32)
                ],
                axis: 0
            )

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else {
            textIds = try tokenize(workingText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])
            audioFeat = MLXArray.zeros([1, textLength, args.patchSize, latentDim])
            textMask = MLXArray.ones([1, textLength], dtype: .float32)
            audioMask = MLXArray.zeros([1, textLength], dtype: .float32)
        }

        let textTokenB = textToken
        let audioFeatB = audioFeat
        let textMaskB = textMask.shape.count == 1 ? textMask.reshaped([1, textMask.dim(0)]) : textMask
        let audioMaskB = audioMask.shape.count == 1 ? audioMask.reshaped([1, audioMask.dim(0)]) : audioMask
        let textMask3 = textMaskB.expandedDimensions(axis: 2)
        let audioMask3 = audioMaskB.expandedDimensions(axis: 2)

        let featEmbed = enc_to_lm_proj(feat_encoder(audioFeatB))
        let textEmbed = base_lm.embedTokens!(textTokenB) * MLXArray(scaleEmb)
        let combinedEmbed = textMask3 * textEmbed + audioMask3 * featEmbed

        let lastFeatIndex = audioFeatB.dim(1) - 1
        var prefixFeatCond = audioFeatB[
            0...,
            lastFeatIndex...(lastFeatIndex),
            0...,
            0...
        ].squeezed(axis: 1)

        let (encOutputs, initialLmCache) = base_lm(inputsEmbeds: combinedEmbed)
        var lmCache = initialLmCache
        let encOutputsFSQ = fsq_layer(encOutputs)
        let maskedEnc = encOutputsFSQ * audioMask3 + encOutputs * textMask3
        var lmHidden = maskedEnc[
            0...,
            (maskedEnc.dim(1) - 1)...(maskedEnc.dim(1) - 1),
            0...
        ].squeezed(axis: 1)

        let residualInput = fusion_concat_proj(
            concatenated([maskedEnc, audioMask3 * featEmbed], axis: -1)
        )
        let (resOutputs, initialResCache) = residual_lm(inputsEmbeds: residualInput)
        var resCache = initialResCache
        var residualHidden = resOutputs[
            0...,
            (resOutputs.dim(1) - 1)...(resOutputs.dim(1) - 1),
            0...
        ].squeezed(axis: 1)

        let hasContinuation = hasPrompt
        var predFeatSeq: [MLXArray] = []
        if hasContinuation {
            let audioIndices = (0..<audioMaskB.dim(1)).filter { idx in
                audioMaskB[0, idx].item(Float.self) > 0.5
            }
            let contextLen = min(streamingPrefixLen - 1, audioIndices.count)
            for idx in audioIndices.suffix(contextLen) {
                let slice = audioFeatB[
                    0...,
                    idx..<(idx + 1),
                    0...,
                    0...
                ]
                predFeatSeq.append(slice)
            }
        }

        let warmupCount = hasContinuation ? 0 : effectiveWarmup
        for step in 0..<(maxTokens + warmupCount) {
            let ditMu = concatenated([
                lm_to_dit_proj(lmHidden),
                res_to_dit_proj(residualHidden)
            ], axis: -1)
            let condIn = prefixFeatCond.transposed(0, 2, 1)

            var predFeat = feat_decoder.sample(
                mu: ditMu,
                nTimesteps: inferenceTimesteps,
                patchSize: args.patchSize,
                cond: condIn,
                cfgValue: cfgValue
            )

            predFeat = predFeat.transposed(0, 2, 1)

            if step >= warmupCount {
                predFeatSeq.append(predFeat.expandedDimensions(axis: 1))
            }

            let currEmbed = enc_to_lm_proj(
                feat_encoder(predFeat.expandedDimensions(axis: 1))
            )

            let stopLogits = stop_head(silu(stop_proj(lmHidden)))
            let stopFlag = argMax(stopLogits, axis: -1).squeezed().item(Int32.self)
            let realSteps = step - warmupCount
            if realSteps > minTokens && stopFlag == 1 {
                break
            }

            let (newLmOut, nextLmCache) = base_lm(
                inputsEmbeds: currEmbed,
                cache: lmCache
            )
            lmCache = nextLmCache
            lmHidden = fsq_layer(newLmOut[
                0...,
                (newLmOut.dim(1) - 1)...(newLmOut.dim(1) - 1),
                0...
            ].squeezed(axis: 1))

            let currResidualInput = fusion_concat_proj(
                concatenated([lmHidden.expandedDimensions(axis: 1), currEmbed], axis: -1)
            )
            let (newResOut, nextResCache) = residual_lm(
                inputsEmbeds: currResidualInput,
                cache: resCache
            )
            resCache = nextResCache
            residualHidden = newResOut[
                0...,
                (newResOut.dim(1) - 1)...(newResOut.dim(1) - 1),
                0...
            ].squeezed(axis: 1)

            prefixFeatCond = predFeat
        }

        guard !predFeatSeq.isEmpty else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "No audio patches were generated"
            ])
        }

        var allFeat = concatenated(predFeatSeq, axis: 1)
        allFeat = allFeat.reshaped([allFeat.dim(0), -1, args.featDim])

        var audio = audio_vae.decode(allFeat)
        audio = audio.flattened()

        if hasContinuation {
            let decodePatchLen = args.patchSize * audio_vae.decodeChunkSize
            let trimAudioSamples = decodePatchLen * (streamingPrefixLen - 1)
            if trimAudioSamples < audio.count {
                audio = audio[trimAudioSamples...]
            }
        }

        eval(audio)
        return audio.asArray(Float.self)
    }
}

extension VoxCPM2TTSModel: SpeechGenerationModel {
    public var sampleRate: Int { outputSampleRate }
}

extension VoxCPM2TTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        base_lm.clearParameters()
        residual_lm.clearParameters()
        feat_encoder.clearParameters()
        feat_decoder.clearParameters()
        fsq_layer.clearParameters()
        enc_to_lm_proj.clearParameters()
        lm_to_dit_proj.clearParameters()
        res_to_dit_proj.clearParameters()
        fusion_concat_proj.clearParameters()
        stop_proj.clearParameters()
        stop_head.clearParameters()
        audio_vae.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return base_lm.parameterMemoryBytes()
            + residual_lm.parameterMemoryBytes()
            + feat_encoder.parameterMemoryBytes()
            + feat_decoder.parameterMemoryBytes()
            + fsq_layer.parameterMemoryBytes()
            + enc_to_lm_proj.parameterMemoryBytes()
            + lm_to_dit_proj.parameterMemoryBytes()
            + res_to_dit_proj.parameterMemoryBytes()
            + fusion_concat_proj.parameterMemoryBytes()
            + stop_proj.parameterMemoryBytes()
            + stop_head.parameterMemoryBytes()
            + audio_vae.parameterMemoryBytes()
    }
}
