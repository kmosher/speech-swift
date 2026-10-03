.PHONY: build debug test clean install sign-installed

CONFIG ?= release
# Install root for fork-built binaries. /opt/ai-tools is provisioned by
# the nix-darwin module of the same name (chown'd to the user, so the
# copy below needs no sudo).
PREFIX ?= /opt/ai-tools

# Code signing. Ad-hoc signatures change on every rebuild, so TCC would treat
# each install as a new program and re-prompt for file and microphone access.
# metawork's `dev-codesign` signs with the shared self-signed identity under
# this program's own identifier, which survives rebuilds; it warns and leaves
# the binaries ad-hoc when the identity was never created.
SIGN_IDENTIFIER ?= dev.kmosher.speech-server
DEV_CODESIGN    ?= /opt/metawork/bin/dev-codesign

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
	@$(MAKE) --no-print-directory sign-installed
	@echo "installed audio-server to $(PREFIX)/bin/"

# Sign in place, after the copy: signing then copying works, but any later
# write to the file invalidates the signature, and `install` writes the
# metallib alongside.
sign-installed:
	@if [ -x "$(DEV_CODESIGN)" ]; then \
		$(DEV_CODESIGN) sign --runtime --identifier $(SIGN_IDENTIFIER) \
			$(PREFIX)/bin/speech-server $(PREFIX)/bin/audio-server; \
	 else \
		echo "warning: $(DEV_CODESIGN) not found — installing ad-hoc."; \
		echo "         TCC will re-prompt for access after every install."; \
	fi

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|MemoryManagementTests|CosyVoiceMemoryTests|SpeakerEncoderUnitTests|PCMConversionTests|ResampleTests|FormatJSONTests|RealtimeAPITests"

clean:
	swift package clean
