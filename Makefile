.PHONY: build debug test clean install

CONFIG ?= release
# Install root for fork-built binaries. /opt/ai-tools is provisioned by
# the nix-darwin module of the same name (chown'd to the user, so the
# copy below needs no sudo).
PREFIX ?= /opt/ai-tools

build:
	swift build -c release --disable-sandbox
	./scripts/build_mlx_metallib.sh release

# Install the freshly-built release binaries + their metallib into PREFIX.
# Symlinks are used so a re-`make install` after a rebuild picks up the new
# binary without having to re-run install. The mlx.metallib needs to sit
# next to the binary at runtime — MLX resolves it by Bundle-style lookup.
install: build
	@mkdir -p $(PREFIX)/bin $(PREFIX)/share/speech-swift
	@cp -f .build/release/audio-server $(PREFIX)/bin/audio-server
	@cp -f .build/release/speech-server $(PREFIX)/bin/speech-server 2>/dev/null || true
	@cp -f .build/arm64-apple-macosx/release/mlx.metallib $(PREFIX)/bin/mlx.metallib
	@echo "installed audio-server to $(PREFIX)/bin/"

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|MemoryManagementTests|CosyVoiceMemoryTests|SpeakerEncoderUnitTests|PCMConversionTests|ResampleTests|FormatJSONTests|RealtimeAPITests"

clean:
	swift package clean
