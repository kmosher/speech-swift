.PHONY: build debug test clean install

CONFIG ?= release
# Install root for fork-built binaries. /opt/ai-tools is provisioned by
# the nix-darwin module of the same name (chown'd to the user, so the
# copy below needs no sudo).
PREFIX ?= /opt/ai-tools

build:
	swift build -c release --disable-sandbox
	./scripts/build_mlx_metallib.sh release

# Install whatever's currently in .build/release/ into PREFIX. Decoupled
# from `build` on purpose: the metallib step requires xcrun's `metal`,
# which isn't always available (broken CLT, etc.). Run `make build`
# explicitly when you want a fresh compile; this target just deploys.
install:
	@test -f .build/release/audio-server || { echo "no .build/release/audio-server — run 'make build' first"; exit 1; }
	@mkdir -p $(PREFIX)/bin $(PREFIX)/share/speech-swift
	@cp -f .build/release/audio-server $(PREFIX)/bin/audio-server
	@cp -f .build/release/speech-server $(PREFIX)/bin/speech-server 2>/dev/null || true
# The metallib is only rebuilt when the Metal toolchain is present, and the
# built path may be a symlink that has gone dangling. Overwrite the installed
# copy only from a real readable file — clobbering a working metallib with
# nothing leaves a server that starts, answers /health, and dies at the first
# inference with "Failed to load the default metallib".
	@if [ -r .build/release/mlx.metallib ]; then 		cp -fL .build/release/mlx.metallib $(PREFIX)/bin/mlx.metallib; 		echo "installed mlx.metallib"; 	elif [ -f $(PREFIX)/bin/mlx.metallib ]; then 		echo "note: no freshly built mlx.metallib; keeping $(PREFIX)/bin/mlx.metallib"; 	else 		echo "error: no mlx.metallib here or in $(PREFIX)/bin — run 'make build' with the Metal toolchain installed" >&2; 		exit 1; 	fi
	@echo "installed audio-server to $(PREFIX)/bin/"

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|MemoryManagementTests|CosyVoiceMemoryTests|SpeakerEncoderUnitTests|PCMConversionTests|ResampleTests|FormatJSONTests|RealtimeAPITests"

clean:
	swift package clean
