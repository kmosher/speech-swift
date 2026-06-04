import Foundation
import Hummingbird
import NIOCore
import ParakeetASR

// MARK: - OpenAI-Compatible Transcription Route
//
// `POST /v1/audio/transcriptions` in the shape OpenAI's Whisper HTTP API
// documents:
//   https://platform.openai.com/docs/api-reference/audio/createTranscription
//
// Body is multipart/form-data with at least a `file` part containing the
// audio (WAV/M4A/MP3/FLAC — anything AVAudioFile reads via AudioFileLoader).
// Other documented form fields (`model`, `language`, `prompt`, `temperature`,
// `response_format`) are accepted; only `response_format` actually changes
// the response shape — the rest are validated-shape only.
//
// Engine: ParakeetASR (CoreML INT8 on ANE). Single model resident across
// requests; first request pays the load cost.

private let parakeetCache = ParakeetCache()

actor ParakeetCache {
    private var model: ParakeetASRModel?

    func load() async throws -> ParakeetASRModel {
        if let m = model { return m }
        let m = try await ParakeetASRModel.fromPretrained()
        model = m
        return m
    }

    /// Drop the resident model. Returns whether one was loaded. The CoreML
    /// model is freed by ARC as its reference clears here.
    func evict() -> Bool {
        let had = model != nil
        model = nil
        return had
    }
}

/// Release the resident Parakeet model (module-internal hook for the idle
/// monitor; the cache instance itself is file-private).
func evictASRModels() async -> Bool {
    await parakeetCache.evict()
}

func registerOpenAITranscribeRoute(on router: Router<BasicRequestContext>) {
    router.post("/v1/audio/transcriptions") { request, _ in
        await activityClock.stamp()
        let contentType = request.headers[.contentType] ?? ""
        guard let boundary = parseMultipartBoundary(contentType) else {
            return errorResponse(
                "Content-Type must be multipart/form-data with a boundary",
                status: .badRequest)
        }
        let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
        let parts = parseMultipart(body: Data(buffer: body), boundary: boundary)

        guard let fileBytes = parts["file"], !fileBytes.isEmpty else {
            return errorResponse(
                "'file' part is required (audio data)",
                status: .badRequest)
        }

        let responseFormat = parts["response_format"]
            .flatMap { String(data: $0, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            ?? "json"
        guard ["json", "text", "verbose_json"].contains(responseFormat) else {
            return errorResponse(
                "'response_format' must be one of: json, text, verbose_json (srt/vtt not supported)",
                status: .badRequest)
        }

        let language = parts["language"]
            .flatMap { String(data: $0, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) }

        // AudioFileLoader handles WAV/M4A/MP3/FLAC/etc via AVAudioFile.
        // Parakeet wants 16kHz mono float.
        let audio: [Float]
        do {
            audio = try decodeWAVData(fileBytes, targetSampleRate: 16000)
        } catch {
            return errorResponse(
                "Failed to decode audio: \(error.localizedDescription)",
                status: .badRequest)
        }

        let model = try await parakeetCache.load()
        let text = try model.transcribeAudio(audio, sampleRate: 16000, language: language)

        switch responseFormat {
        case "text":
            return Response(
                status: .ok,
                headers: [.contentType: "text/plain; charset=utf-8"],
                body: .init(byteBuffer: .init(string: text)))
        case "verbose_json":
            let duration = Double(audio.count) / 16000.0
            return jsonResponse([
                "task": "transcribe",
                "language": language ?? "en",
                "duration": round(duration * 100) / 100,
                "text": text
            ] as [String: Any])
        default:  // "json"
            return jsonResponse(["text": text])
        }
    }
}

// MARK: - Multipart/form-data parsing
//
// Minimal RFC 7578 parser. Handles named parts only; filename and per-part
// Content-Type are read but unused. Sufficient for OpenAI's audio API shape
// where every field is name-keyed.

/// Extract `boundary=...` from a Content-Type header like
/// `multipart/form-data; boundary=----WebKitFormBoundaryXYZ`.
func parseMultipartBoundary(_ contentType: String) -> String? {
    guard contentType.lowercased().contains("multipart/form-data") else { return nil }
    for raw in contentType.split(separator: ";") {
        let piece = raw.trimmingCharacters(in: .whitespaces)
        if piece.lowercased().hasPrefix("boundary=") {
            var b = String(piece.dropFirst("boundary=".count))
            // Strip optional surrounding quotes.
            if b.hasPrefix("\""), b.hasSuffix("\""), b.count >= 2 {
                b = String(b.dropFirst().dropLast())
            }
            return b.isEmpty ? nil : b
        }
    }
    return nil
}

/// Parse a multipart body into `name -> raw bytes`. Returns an empty dict on
/// malformed input rather than throwing — callers validate required fields.
func parseMultipart(body: Data, boundary: String) -> [String: Data] {
    var out: [String: Data] = [:]
    guard let delimiter = "--\(boundary)".data(using: .utf8),
          let crlfCrlf = "\r\n\r\n".data(using: .utf8),
          let crlf = "\r\n".data(using: .utf8) else {
        return out
    }
    // Split body on boundary delimiter. Each segment is a part (except the
    // first chunk before the first delimiter and the final `--boundary--`
    // closer, which we filter).
    var cursor = 0
    var segments: [Data] = []
    while cursor < body.count {
        guard let range = body.range(of: delimiter, in: cursor..<body.count) else {
            break
        }
        if cursor < range.lowerBound {
            segments.append(body.subdata(in: cursor..<range.lowerBound))
        }
        cursor = range.upperBound
    }

    for var segment in segments {
        // Skip leading CRLF after the delimiter, and trailing CRLF before the
        // next delimiter. The closing boundary segment is "--\r\n" — has no
        // headers/body, so the header-block search below fails and we skip.
        if segment.starts(with: crlf) {
            segment = segment.subdata(in: crlf.count..<segment.count)
        }
        if segment.suffix(crlf.count) == crlf {
            segment = segment.subdata(in: 0..<(segment.count - crlf.count))
        }
        guard let split = segment.range(of: crlfCrlf) else { continue }
        let headerData = segment.subdata(in: 0..<split.lowerBound)
        let bodyData = segment.subdata(in: split.upperBound..<segment.count)
        guard let headerStr = String(data: headerData, encoding: .utf8) else { continue }
        guard let name = parseFormDataName(headerStr) else { continue }
        out[name] = bodyData
    }
    return out
}

/// Pull the `name="..."` value from a `Content-Disposition: form-data; ...`
/// header block. Case-insensitive on the header name; quote-stripped on the value.
private func parseFormDataName(_ headers: String) -> String? {
    for line in headers.split(separator: "\r\n") {
        let lower = line.lowercased()
        guard lower.hasPrefix("content-disposition:") else { continue }
        // Find name="..."
        let nameKey = "name=\""
        guard let nameRange = line.range(of: nameKey) else { continue }
        let afterName = line[nameRange.upperBound...]
        guard let endQuote = afterName.firstIndex(of: "\"") else { continue }
        return String(afterName[..<endQuote])
    }
    return nil
}
