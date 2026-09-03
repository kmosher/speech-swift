import Foundation
import Hummingbird

// MARK: - TTS engine registry
//
// Adapted from upstream's ModelRegistry, whose promise is the one worth
// keeping: adding an engine is one row, with no new dispatch arm and no new
// resolver. The shape differs because this route selects differently — upstream
// resolves the `model` field to a variant, while here `model` is unreliable
// (voicemode stamps its legacy VoxCPM2 default on every request) and the
// engine is chosen from the `voice` field's prefix instead.
//
// Before this, each engine added an arm to a `hasPrefix` chain and a matching
// branch in two separate places — the explicit-clone_ref path and the registry
// path — which is two chances to wire an engine into one and forget the other.

/// One selectable synthesis engine.
///
/// `clone` is the only required behavior: every engine on this route renders a
/// specific voice from a reference. `bare` is optional because most engines
/// have no meaningful default speaker — asking for one gets you a voice nobody
/// on this machine wants.
struct TTSEngine: Sendable {
    /// Lowercased prefix matched against the `voice` field. Empty means this
    /// engine is the fallback for a voice that names no engine at all.
    let selector: String

    /// Human name, for logs and errors.
    let name: String

    /// Whether this engine consumes the reference transcript. CosyVoice
    /// conditions on a speaker embedding and has nowhere to put one, so
    /// sending it would be silently ignored rather than merely redundant.
    let usesReferenceText: Bool

    /// Render `input` in the voice of `cloneRef`.
    let clone: @Sendable (_ input: String, _ cloneRef: String, _ cloneRefText: String?,
                          _ responseFormat: String, _ instructions: String?,
                          _ modelId: String?, _ speed: Double) async throws -> Response

    /// Render `input` in the engine's own default speaker, when it has one.
    let bare: (@Sendable (_ input: String, _ responseFormat: String,
                          _ instructions: String?, _ modelId: String?) async throws -> Response)?
}

/// Every engine reachable on `/v1/audio/speech`, in match order.
///
/// The empty selector must stay last: it is the fallback, and a prefix match
/// against "" succeeds for every voice.
let TTS_ENGINES: [TTSEngine] = [
    TTSEngine(
        selector: "voxcpm2",
        name: "VoxCPM2",
        usesReferenceText: true,
        clone: { input, ref, refText, format, instructions, modelId, _ in
            try await handleVoxCPM2Clone(
                input: input, cloneRef: ref, cloneRefText: refText,
                responseFormat: format, instructions: instructions, modelId: modelId)
        },
        bare: { input, format, instructions, modelId in
            try await handleVoxCPM2Bare(
                input: input, responseFormat: format,
                instructions: instructions, modelId: modelId)
        }),
    TTSEngine(
        selector: "cosyvoice",
        name: "CosyVoice",
        usesReferenceText: false,
        clone: { input, ref, _, format, _, modelId, _ in
            try await handleCosyVoiceClone(
                input: input, cloneRef: ref, responseFormat: format, modelId: modelId)
        },
        bare: nil),
    TTSEngine(
        selector: "",
        name: "F5-TTS",
        usesReferenceText: true,
        clone: { input, ref, refText, format, _, _, speed in
            try await handleF5Clone(
                input: input, cloneRef: ref, cloneRefText: refText,
                responseFormat: format, speed: speed)
        },
        // F5 has no bare mode; a voice that names no engine and no reference
        // falls through to VoxCPM2's default speaker.
        bare: nil),
]

/// The engine a `voice` value selects.
///
/// Always returns something: the empty-selector row matches anything, which is
/// what makes F5 the default for registry and clone_ref voices without a
/// special case for "no engine named".
func resolveTTSEngine(voice: String) -> TTSEngine {
    let lower = voice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    for engine in TTS_ENGINES where !engine.selector.isEmpty {
        if lower.hasPrefix(engine.selector) { return engine }
    }
    return TTS_ENGINES[TTS_ENGINES.count - 1]
}

/// The engine used when a request names neither an engine nor a reference.
///
/// Separate from `resolveTTSEngine` because the fallback for *selection* (F5,
/// which can only clone) and the fallback for *having nothing to clone from*
/// (VoxCPM2's default speaker) are different answers to different questions.
func bareTTSEngine() -> TTSEngine? {
    TTS_ENGINES.first { $0.bare != nil }
}
