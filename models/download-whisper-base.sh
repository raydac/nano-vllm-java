#!/usr/bin/env bash
# Download openai/whisper-base as a Hugging Face Whisper safetensors folder.
# Speech-to-text (audio -> text). Not a chat model. Not CTranslate2 model.bin.
# Source: https://huggingface.co/openai/whisper-base
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/whisper-base"
BASE="https://huggingface.co/openai/whisper-base/resolve/main"

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
  if ! curl_resume "$f" "$BASE/$f" "${AUTH[@]}"; then
    echo "Download failed for $f." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Model card: https://huggingface.co/openai/whisper-base" >&2
    exit 1
  fi
}

echo "Downloading config / tokenizer sidecars ..."
for f in config.json tokenizer.json tokenizer_config.json; do
  download "$f"
done

echo "Downloading model.safetensors (~290MB) ..."
download model.safetensors

echo "Installed to $DEST"
ls -lh
du -sh .
echo "Note: Whisper speech-to-text (HF safetensors). Do not use a faster-whisper model.bin folder."
echo "Load: LlmModelFactory.make(Path.of(\"models/whisper-base\")); model.transcribe(wavPath)"
