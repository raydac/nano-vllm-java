#!/usr/bin/env bash
# Download Tiny Voca English (config, f16 GGUF, pronunciation lexicon).
# Text-to-speech (text -> WAV). Not a chat model. Not ONNX Runtime.
# https://huggingface.co/n4ze3m/tiny-voca
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/tiny-voca"
BASE="https://huggingface.co/n4ze3m/tiny-voca/resolve/main"

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
elif [[ -n "${HF_HOME:-}" && -f "${HF_HOME}/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HF_HOME}/token")")
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HOME}/.cache/huggingface/token")")
fi

mkdir -p "$DEST/phonemizer"

download() {
  local dest="$1"
  local remote="$2"
  echo "Downloading $remote ..."
  if ! curl_resume "$dest" "$BASE/$remote" -A "Mozilla/5.0" "${AUTH[@]}"; then
    echo "Download failed for $remote." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Model card: https://huggingface.co/n4ze3m/tiny-voca" >&2
    exit 1
  fi
}

download "$DEST/config.json" config.json
download "$DEST/tiny-voca-f16.gguf" gguf/tiny-voca-f16.gguf
download "$DEST/phonemizer/lexicon.tsv" phonemizer/lexicon.tsv

echo "Installed to $DEST"
ls -lh "$DEST" "$DEST/phonemizer"
du -sh "$DEST"
echo "Load: LlmModelFactory.open(Path.of(\"models/tiny-voca\")).make()"
echo "Try:  mvn -pl nano-vllm-java-samples -q exec:java -Dexec.args=models/tiny-voca"
echo "Voice: type  voice=F2 Hello   or  voice=M2 Hello   at the tts?> prompt."
