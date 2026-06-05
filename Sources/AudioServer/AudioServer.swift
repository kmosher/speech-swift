import Foundation
import Hummingbird
import HummingbirdCore
import MLX
import NIOCore
import AudioCommon

// MARK: - Server
//
// Trimmed audio-server: serves only OpenAI-compatible REST endpoints.
//   GET  /health
//   POST /v1/audio/speech          — VoxCPM2 TTS (Routes+OpenAI.swift)
//   POST /v1/audio/transcriptions  — Parakeet ASR (Routes+OpenAITranscribe.swift)
//
// Legacy native routes (/transcribe, /speak, /respond, /enhance) and the
// /v1/realtime WS upgrade were removed 2026-06-02 — voicemode is the only
// client and only uses the OpenAI paths. PersonaPlex / SpeechEnhancement /
// Qwen3-ASR/-TTS / CosyVoice modules are still in the workspace; this binary
// just doesn't link them.

public struct AudioServer {
    let host: String
    let port: Int
    /// Seconds of inactivity after which resident models are released. 0
    /// disables eviction (models stay resident for the life of the process).
    let idleTimeout: Double

    public init(host: String = "127.0.0.1", port: Int = 8080, preload: Bool = false, idleTimeout: Double = 0) {
        self.host = host
        self.port = port
        self.idleTimeout = idleTimeout
    }

    public func run() async throws {
        // Bound MLX's reusable buffer cache. With no limit it grows to the
        // high-water mark of every distinct allocation size seen over the
        // process lifetime; F5's per-request, duration-dependent buffer sizes
        // made that balloon to ~50GB across a session. 4GB keeps buffer reuse
        // fast while returning the rest to the OS. The idle monitor still does a
        // full clearCache() after inactivity; this just caps growth while busy.
        MLX.Memory.cacheLimit = 4 * 1024 * 1024 * 1024

        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port)))
        if idleTimeout > 0 {
            Task { await runIdleMonitor(timeout: idleTimeout) }
        }
        try await app.run()
    }

    public func preloadModels() async throws {
        // Models load lazily on first request. Kept as a no-op so the CLI's
        // --preload flag remains a valid argument; a future implementation
        // can warm Parakeet + VoxCPM2 here.
    }

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router<BasicRequestContext>()

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: "{\"status\":\"ok\"}")))
        }

        registerOpenAIRoutes(on: router)
        registerOpenAITranscribeRoute(on: router)

        return router
    }
}

// MARK: - Audio Encoding/Decoding

func decodeWAVData(_ data: Data, targetSampleRate: Int) throws -> [Float] {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try data.write(to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try AudioFileLoader.load(url: tmpURL, targetSampleRate: targetSampleRate)
}

func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try Data(contentsOf: tmpURL)
}

// MARK: - Response Helpers

func jsonResponse(_ dict: [String: Any]) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])) ?? Data()
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

func errorResponse(_ message: String, status: HTTPResponse.Status) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: ["error": message], options: [])) ?? Data()
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}
