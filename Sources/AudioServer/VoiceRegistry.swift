import CryptoKit
import Foundation

/// On-disk registry of pre-curated voice-clone references. The audio-server
/// scans `~/.voicemode/voices/<id>/` (override via `VOICEMODE_VOICES_DIR`) at
/// startup; each subdirectory must contain `ref.wav` and `ref.txt`. The id is
/// the directory name.
///
/// Two ways callers select a registry voice via the `voice` field on
/// `/v1/audio/speech`:
///   1. Bare id, e.g. `"ed_irish_m"` — exact match.
///   2. `claude_<hex>` / `blend_<hex>` — SHA-256 hashed across the sorted id
///      list. Same per-session id always hashes to the same registry entry,
///      so multiple Claude sessions get audibly distinct voices for free.
///
/// When the registry is non-empty the claude_/blend_ hash hits it instead of
/// the old Qwen3 CustomVoice 3-bucket. Empty registry → old behavior.
struct VoiceEntry: Sendable {
    let id: String
    let refPath: String
    let refText: String
}

final class VoiceRegistry: @unchecked Sendable {
    private let entries: [String: VoiceEntry]
    private let sortedIds: [String]

    init(directory: String? = nil) {
        let dir = directory ?? Self.defaultDirectory()
        var collected: [String: VoiceEntry] = [:]
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: dir) {
            for name in contents {
                let entryDir = (dir as NSString).appendingPathComponent(name)
                var isDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: entryDir, isDirectory: &isDir),
                      isDir.boolValue else { continue }
                let refWav = (entryDir as NSString).appendingPathComponent("ref.wav")
                let refTxt = (entryDir as NSString).appendingPathComponent("ref.txt")
                guard FileManager.default.fileExists(atPath: refWav),
                      let text = try? String(contentsOfFile: refTxt, encoding: .utf8) else {
                    continue
                }
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                collected[name] = VoiceEntry(id: name, refPath: refWav, refText: trimmed)
            }
        }
        self.entries = collected
        self.sortedIds = collected.keys.sorted()
        if !sortedIds.isEmpty {
            let joined = sortedIds.joined(separator: ", ")
            let msg = "[voice-registry] loaded \(sortedIds.count) voices from \(dir): \(joined)\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
    }

    var isEmpty: Bool { sortedIds.isEmpty }

    func lookup(id: String) -> VoiceEntry? {
        entries[id]
    }

    /// Hash a `claude_<hex>` / `blend_<hex>` session id to a stable registry
    /// entry. Returns nil if the registry is empty.
    func hashedLookup(sessionId: String) -> VoiceEntry? {
        guard !sortedIds.isEmpty else { return nil }
        let digest = Array(SHA256.hash(data: Data(sessionId.utf8)))
        var idx: UInt64 = 0
        for byte in digest.prefix(8) {
            idx = (idx << 8) | UInt64(byte)
        }
        let pick = sortedIds[Int(idx % UInt64(sortedIds.count))]
        return entries[pick]
    }

    private static func defaultDirectory() -> String {
        if let override = ProcessInfo.processInfo.environment["VOICEMODE_VOICES_DIR"],
           !override.isEmpty {
            return override
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return (home as NSString).appendingPathComponent(".voicemode/voices")
    }
}

/// Process-wide singleton, loaded once at first reference.
let voiceRegistry = VoiceRegistry()
