#!/usr/bin/env bash
# Download Piper Russian Irina medium (ONNX + sidecar) and espeak-ng-data.
# Text-to-speech (text -> WAV). Not a chat model. Not ONNX Runtime.
# Voice: https://huggingface.co/rhasspy/piper-voices
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
# shellcheck source=./_espeak_ng_data.sh
. "$MODELS_ROOT/_espeak_ng_data.sh"
DEST="$MODELS_ROOT/piper-ru-irina-medium"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium"

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
elif [[ -n "${HF_HOME:-}" && -f "${HF_HOME}/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HF_HOME}/token")")
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HOME}/.cache/huggingface/token")")
fi

mkdir -p "$DEST"
cd "$DEST"

download() {
  local f="$1"
  echo "Downloading $f ..."
  if ! curl_resume "$f" "$VOICE_BASE/$f" "${AUTH[@]}"; then
    echo "Download failed for $f." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Voice card: https://huggingface.co/rhasspy/piper-voices" >&2
    exit 1
  fi
}

echo "Downloading Piper Irina medium ONNX + sidecar ..."
download ru_RU-irina-medium.onnx
download ru_RU-irina-medium.onnx.json

install_espeak_ng_data "$DEST"

echo "Installed to $DEST"
ls -lh
du -sh .
echo "Load: LlmModelFactory.open(Path.of(\"models/piper-ru-irina-medium\"))"
echo "      .optionalData(LlmOptionalData.ESPEAK_DATA, Path.of(\"models/piper-ru-irina-medium/espeak-ng-data\"))"
echo "      .make(); model.synthesize(\"Привет, мир\")"
