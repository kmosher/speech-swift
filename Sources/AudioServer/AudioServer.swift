import Foundation
import Hummingbird
import HummingbirdCore
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

    public init(host: String = "127.0.0.1", port: Int = 8080, preload: Bool = false) {
        self.host = host
        self.port = port
    }

    public func run() async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port)))
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
