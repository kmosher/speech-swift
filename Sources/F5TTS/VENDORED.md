# Vendored sources

`Sources/F5TTS` and `Sources/Vocos` are vendored (copied in-tree) from two
MIT-licensed packages by Lucas Newman, rather than consumed as SwiftPM
dependencies:

| dir            | upstream                                  | commit    |
|----------------|-------------------------------------------|-----------|
| `Sources/F5TTS`| https://github.com/lucasnewman/f5-tts-swift | `36899100` |
| `Sources/Vocos`| https://github.com/lucasnewman/vocos-swift  | `f9502c7f` |

**Why in-tree:** `f5-tts-swift`'s manifest pins `swift-transformers < 1.0`,
which conflicts irreconcilably with this repo's `1.1.x` requirement (and its
`mlx-swift 0.18` floor vs. our `0.30`). The *source* already compiles cleanly
against our resolved MLX 0.30 / transformers 1.1.6 — only the upstream
`Package.swift` version pins were the obstacle — so vendoring the sources as
plain targets of this package sidesteps the resolution conflict entirely. This
matches how the other TTS engines here (VoxCPM2, Kokoro, CosyVoice) already live
in-tree.

**Local modifications** (see `git log` / `git blame` for specifics):
- `F5TTS.swift`: cache the Vocos vocoder on the instance instead of reloading it
  from disk every `generate()` call; add an array-based `generate(referenceAudio:)`
  overload so the server can pass a pre-resampled 24kHz reference (avoids the
  chipmunk trap from `AudioUtilities.loadAudioFile`, which does not resample).
- `F5TTS.swift`: relax `update(verify:)` from `.all` to `.noUnusedKeys` — the
  precomputed rotary `freqsCis` buffers are intentionally absent from the
  checkpoint, which `.all` rejects under MLX 0.30's parameter reflection.

Upstream MIT license text is reproduced in `LICENSE` at the repo root context;
copyright remains © 2024 Lucas Newman.
