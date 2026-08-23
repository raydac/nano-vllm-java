#!/usr/bin/env bash
# Download Gemma 4 E2B IT QAT mobile (wNa8o8) into models/Gemma4-E2B-IT-QAT-Mobile.
# Apache 2.0, ungated. Optional HF_TOKEN if Hugging Face rate-limits you.
#   https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/Gemma4-E2B-IT-QAT-Mobile"
BASE="https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main"

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
    echo "Model card: https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers" >&2
    exit 1
  fi
}

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         chat_template.jinja preprocessor_config.json processor_config.json; do
  download "$f"
done

echo "Downloading model.safetensors (~2.3GB) ..."
download model.safetensors

echo "Installed to $DEST"
ls -lh
du -sh .
