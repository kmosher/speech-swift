# speech-server: build, install, and lifecycle (voicemode backend)

`speech-server` is the single binary this fork ships for [voicemode](https://github.com/kmosher/voicemode):
VoxCPM2 TTS (`/v1/audio/speech`) + Parakeet STT (`/v1/audio/transcriptions`) on
one port (default `:8893`), OpenAI-compatible. voicemode is the only client.

## Building

**Standard path** (needs full Xcode + the Metal Toolchain):

```bash
make build          # compiles the Swift package AND mlx.metallib
```

`make build` (or `./scripts/build_mlx_metallib.sh release` after a manual
`swift build`) compiles MLX's Metal kernels into `mlx.metallib`. That step
shells out to `xcrun -sdk macosx metal`/`metallib`, which live in the **Metal
Toolchain** — present in full Xcode, installable via
`xcodebuild -downloadComponent MetalToolchain`.

**CLT-only fallback** (Command Line Tools, no Metal Toolchain):

On a machine with only Command Line Tools selected (`xcode-select -p` →
`.../CommandLineTools`), `xcrun --find metal` fails and you **cannot** compile
the metallib. But you don't have to:

```bash
swift build -c release --product speech-server          # binary only
cp /path/to/existing/mlx.metallib .build/release/        # reuse a prebuilt one
```

The metallib depends **only on the mlx-swift version** (pinned in
`Package.resolved`), not on any Swift source in this repo — so a previously
built `mlx.metallib` is byte-for-byte valid for any rebuild of the server.
Reuse it freely; you only need the Metal Toolchain to regenerate it when you
**bump mlx-swift**. (`scripts/build_mlx_metallib.sh` content-hashes the kernel
sources and skips recompiling when they're unchanged.)

### metallib placement

MLX loads `mlx.metallib` from the **executable's own directory**. Whatever
builds/installs `speech-server` must stage `mlx.metallib` next to it, or the
first inference dies with `Failed to load the default metallib`.

## Install

This fork's binaries live in `/opt/ai-tools/bin/` (kmosher's nix-darwin
`ai-tools.nix` creates and owns it — isolated from Homebrew and `~/.local/bin`).
Install = copy both files there:

```bash
cp .build/release/speech-server /opt/ai-tools/bin/speech-server
cp .build/release/mlx.metallib  /opt/ai-tools/bin/mlx.metallib   # if rebuilt
```

## Idle model eviction

`--idle-timeout <seconds>` makes the server release its resident models
(VoxCPM2 is ~9 GB of MLX weights, plus pooled inference buffers) after that
many seconds with no request, returning MLX's buffer pool to the OS. The HTTP
listener stays bound, so clients never see a refused connection — the next
request reloads lazily (cheap: weights are mmap'd, so first-request latency is
dominated by inference, not load). `0` (default) disables eviction (models stay
resident for the life of the process).

## Lifecycle: let voicemode manage it

voicemode owns the launchd lifecycle as a first-class service (same pattern it
uses for the external `mlx-audio` binary):

```bash
voicemode service enable speech-server      # install + load com.voicemode.speech-server
voicemode service status speech-server
voicemode service {start,stop,restart,disable,logs} speech-server
```

Config is read from `~/.voicemode/voicemode.env` at startup:
`VOICEMODE_SPEECH_SERVER_{BIN,HOST,PORT,IDLE_TIMEOUT}` (defaults:
`/opt/ai-tools/bin/speech-server`, `127.0.0.1`, `8893`, `900`).

**It must run as a LaunchAgent, not a daemon.** MLX needs a Metal device, which
is only available inside the user's Aqua/GUI login session. A daemon (or any
headless/sandboxed launch) gets an empty Metal device list and SIGABRTs on the
first inference.
