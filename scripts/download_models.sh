#!/usr/bin/env bash
# Downloads the open-source speech models used by Buddy (about 400 MB) from GitHub releases.
# Run once on the Raspberry Pi (needs internet). Safe to re-run: existing models are skipped.
set -euo pipefail
DEST="${BUDDY_HOME:-$HOME/buddy-data}/models"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
mkdir -p "$DEST"
cd "$DEST"

get_tar() {  # $1 = release tag, $2 = archive name without .tar.bz2
  if [ -d "$2" ]; then echo "ok   $2"; return; fi
  echo "get  $2"
  curl -fL --retry 3 -o "$2.tar.bz2" "$BASE/$1/$2.tar.bz2"
  tar xf "$2.tar.bz2" && rm -f "$2.tar.bz2"
}

# Speech-to-text: Moonshine v2 English (MIT licence)
get_tar asr-models sherpa-onnx-moonshine-base-en-quantized-2026-02-27
# (optional, lighter/faster) uncomment to also get the tiny model:
# get_tar asr-models sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

# Voice activity detection: Silero VAD (MIT licence)
if [ ! -f silero_vad.onnx ]; then
  echo "get  silero_vad.onnx"
  curl -fL --retry 3 -o silero_vad.onnx "$BASE/asr-models/silero_vad.onnx"
else echo "ok   silero_vad.onnx"; fi

# Text-to-speech: Kokoro-82M int8 (Apache-2.0) + Piper LJSpeech (public-domain data) as fast alternative
get_tar tts-models kokoro-int8-en-v0_19
get_tar tts-models vits-piper-en_US-ljspeech-medium

echo "Models ready in $DEST"
du -sh "$DEST"
