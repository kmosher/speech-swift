.PHONY: build debug test clean install sign-installed

CONFIG ?= release
# Install root for fork-built binaries. /opt/ai-tools is provisioned by
# the nix-darwin module of the same name (chown'd to the user, so the
# copy below needs no sudo).
PREFIX ?= /opt/ai-tools

# Code signing. TCC keys a permission grant on the binary's *designated
# requirement*. Ad-hoc signing has no certificate to anchor to, so that
# requirement degrades to `cdhash H"..."` — a content hash — and every rebuild
# is a different program to TCC, re-prompting for file access on each install.
# Signing with a stable identity makes it `identifier "..." and certificate
# root = H"..."`, which survives rebuilds and keeps the grant.
#
# The identity is the self-signed one tmvault creates (scripts live in that
# repo); a distinct SIGN_IDENTIFIER keeps the two programs separate while
# sharing the certificate. Signing is skipped with a warning when the identity
# is absent, so this stays buildable on a machine that has never had it.
SIGN_IDENTITY   ?= tmvault-signing
SIGN_IDENTIFIER ?= dev.kmosher.speech-server
SIGN_KEYCHAIN   ?= $(HOME)/Library/Keychains/tmvault-signing.keychain-db
SIGN_PASS_FILE  ?= $(HOME)/.config/tmvault/signing-keychain.pass

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
# metallib alongside. `find-identity` is checked without -v because a
# self-signed certificate reads as CSSMERR_TP_NOT_TRUSTED — codesign accepts
# it regardless, and -v would filter out the only identity we have.
sign-installed:
	@if [ -f "$(SIGN_PASS_FILE)" ] && [ -f "$(SIGN_KEYCHAIN)" ]; then 		security unlock-keychain -p "$$(cat $(SIGN_PASS_FILE))" $(SIGN_KEYCHAIN) 2>/dev/null || true; 	fi
	@if security find-identity -p codesigning 2>/dev/null | grep -q "$(SIGN_IDENTITY)"; then 		for bin in speech-server audio-server; do 			codesign --force --options runtime 				--identifier $(SIGN_IDENTIFIER) 				--sign $(SIGN_IDENTITY) --timestamp=none 				$(PREFIX)/bin/$$bin || exit 1; 		done; 		echo "signed as $(SIGN_IDENTIFIER) ($(SIGN_IDENTITY))"; 	else 		echo "warning: signing identity '$(SIGN_IDENTITY)' not found — installing ad-hoc."; 		echo "         TCC will re-prompt for file access after every install."; 	fi

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|MemoryManagementTests|CosyVoiceMemoryTests|SpeakerEncoderUnitTests|PCMConversionTests|ResampleTests|FormatJSONTests|RealtimeAPITests"

clean:
	swift package clean
