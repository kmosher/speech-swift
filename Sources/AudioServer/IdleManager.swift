import Foundation
import MLX

// MARK: - Idle model eviction
//
// The server stays bound to its port for the life of the process, but the
// expensive part — the resident VoxCPM2 (~9 GB of MLX weights) and Parakeet
// models — is released once no request has arrived for `idleTimeout` seconds.
// An idle server then costs essentially nothing (a bare HTTP listener, no GPU
// allocation); the next request reloads lazily. Reload is cheap relative to a
// request anyway: the weights are mmap'd, so first-request latency is
// dominated by inference, not load. Because the listener never goes away,
// clients never see a refused connection — they just occasionally pay one
// reload.

/// Wall-clock of the most recent request. Stamped by every request handler,
/// read by `runIdleMonitor`.
actor ActivityClock {
    private var last = Date()
    func stamp() { last = Date() }
    func idleSeconds() -> Double { Date().timeIntervalSince(last) }
}

let activityClock = ActivityClock()

/// Background loop that releases model memory after `timeout` seconds idle.
/// Runs for the life of the process; callers skip it entirely when
/// `timeout <= 0` (eviction disabled — the pre-existing always-resident
/// behaviour).
func runIdleMonitor(timeout: Double) async {
    // Evict at most once per idle stretch: latch after evicting, and clear the
    // latch only once a request has reset the clock. Without this we'd re-run
    // clearCache() and re-log on every tick while the server sits idle.
    var evictedThisIdle = false
    while true {
        try? await Task.sleep(for: .seconds(30))
        let idle = await activityClock.idleSeconds()
        if idle < timeout {
            evictedThisIdle = false
            continue
        }
        if evictedThisIdle { continue }
        evictedThisIdle = true

        // Snapshot before dropping references: VoxCPM2's MLXArrays return to
        // MLX's buffer pool as they deallocate, and clearCache() then hands
        // that pool back to the OS. (Parakeet is CoreML/ANE, freed by ARC when
        // its reference drops — not reflected in the MLX figure below.)
        let before = MLX.Memory.activeMemory + MLX.Memory.cacheMemory
        let ttsReleased = await evictTTSModels()
        let asrReleased = await evictASRModels()
        if ttsReleased == 0 && !asrReleased { continue }  // nothing was loaded
        MLX.Memory.clearCache()
        let freedMB = Double(before - (MLX.Memory.activeMemory + MLX.Memory.cacheMemory)) / 1_048_576
        let msg = "[idle] released models after \(Int(idle))s idle "
            + "(tts variants: \(ttsReleased), asr: \(asrReleased)); "
            + "reclaimed ~\(String(format: "%.0f", freedMB)) MB of MLX memory\n"
        FileHandle.standardError.write(Data(msg.utf8))
    }
}
