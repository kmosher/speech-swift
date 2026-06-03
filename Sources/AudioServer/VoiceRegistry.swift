import CryptoKit
import Foundation

/// On-disk registry of pre-curated voice-clone references. The audio-server
/// scans `~/.voicemode/voices/<id>/` (override via `VOICEMODE_VOICES_DIR`) at
/// startup; each subdirectory must contain `ref.wav` and `ref.txt`. The id is
/// the directory name. An optional empty `no_hash` file marks the voice as
/// "available via direct lookup only" — excluded from the `claude_<hex>` hash
/// pool. Use this for voices you don't want Claude sessions randomly landing
/// on (e.g. yourself), while still keeping them callable by id.
///
/// Two ways callers select a registry voice via the `voice` field on
/// `/v1/audio/speech`:
///   1. Bare id, e.g. `"ed_irish_m"` — exact match. Honors `no_hash` voices.
///   2. `claude_<hex>` / `blend_<hex>` — SHA-256 hashed across the sorted
///      *hashable* id list (those without `no_hash`). Same per-session id
///      always hashes to the same registry entry.
struct VoiceEntry: Sendable {
    let id: String
    let refPath: String
    let refText: String
    /// When true, excluded from `hashedLookup`; only reachable by bare id.
    let optOutOfHash: Bool
}

final class VoiceRegistry: @unchecked Sendable {
    private let entries: [String: VoiceEntry]
    private let hashableIds: [String]

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
                let noHash = (entryDir as NSString).appendingPathComponent("no_hash")
                guard FileManager.default.fileExists(atPath: refWav),
                      let text = try? String(contentsOfFile: refTxt, encoding: .utf8) else {
                    continue
                }
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                collected[name] = VoiceEntry(
                    id: name,
                    refPath: refWav,
                    refText: trimmed,
                    optOutOfHash: FileManager.default.fileExists(atPath: noHash))
            }
        }
        self.entries = collected
        self.hashableIds = collected.values.filter { !$0.optOutOfHash }.map(\.id).sorted()
        if !collected.isEmpty {
            let allIds = collected.keys.sorted()
            let excluded = allIds.filter { collected[$0]?.optOutOfHash == true }
            var msg = "[voice-registry] loaded \(allIds.count) voices from \(dir): \(allIds.joined(separator: ", "))"
            if !excluded.isEmpty {
                msg += " (no_hash: \(excluded.joined(separator: ", ")))"
            }
            msg += "\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
    }

    var isEmpty: Bool { entries.isEmpty }

    func lookup(id: String) -> VoiceEntry? {
        entries[id]
    }

    /// Hash a `claude_<hex>` / `blend_<hex>` session id to a stable registry
    /// entry. Returns nil if no hashable entries exist. Voices with `no_hash`
    /// are excluded from the pool but still reachable via `lookup(id:)`.
    func hashedLookup(sessionId: String) -> VoiceEntry? {
        guard !hashableIds.isEmpty else { return nil }
        let digest = Array(SHA256.hash(data: Data(sessionId.utf8)))
        var idx: UInt64 = 0
        for byte in digest.prefix(8) {
            idx = (idx << 8) | UInt64(byte)
        }
        let pick = hashableIds[Int(idx % UInt64(hashableIds.count))]
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
