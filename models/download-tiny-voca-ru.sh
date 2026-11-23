#!/usr/bin/env bash
# Download Tiny Voca Russian (config + safetensors). Same acoustic graph as English.
# Text-to-speech (text -> WAV). Not a chat model. Stress defaults to the last vowel
# unless phonemizer/stress.txt sits next to the checkpoint.
# https://huggingface.co/egnch/tiny-voca-ru
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/tiny-voca-ru"
BASE="https://huggingface.co/egnch/tiny-voca-ru/resolve/main"

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
elif [[ -n "${HF_HOME:-}" && -f "${HF_HOME}/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HF_HOME}/token")")
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HOME}/.cache/huggingface/token")")
fi

mkdir -p "$DEST"

download() {
  local dest="$1"
  local remote="$2"
  echo "Downloading $remote ..."
  if ! curl_resume "$dest" "$BASE/$remote" -A "Mozilla/5.0" "${AUTH[@]}"; then
    echo "Download failed for $remote." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Model card: https://huggingface.co/egnch/tiny-voca-ru" >&2
    exit 1
  fi
}

download "$DEST/config.json" config.json
download "$DEST/model.safetensors" model.safetensors

echo "Installed to $DEST"
ls -lh "$DEST"
du -sh "$DEST"
echo "Load: LlmModelFactory.open(Path.of(\"models/tiny-voca-ru\")).make()"
echo "Try:  mvn -pl nano-vllm-java-samples -q exec:java -Dexec.args=models/tiny-voca-ru"
echo "Voice: type  voice=F Привет   or  voice=M Привет   at the tts?> prompt."
