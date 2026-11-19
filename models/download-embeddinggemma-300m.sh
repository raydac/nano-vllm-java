#!/usr/bin/env bash
# Download EmbeddingGemma-300M into models/EmbeddingGemma-300M.
# Requires accepting the Gemma license on Hugging Face and usually HF_TOKEN.
#   huggingface-cli login
#   export HF_TOKEN=hf_...
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/EmbeddingGemma-300M"
BASE="https://huggingface.co/google/embeddinggemma-300m/resolve/main"

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
  mkdir -p "$(dirname "$f")"
  echo "Downloading $f ..."
  if ! curl_resume "$f" "$BASE/$f" "${AUTH[@]}"; then
    echo "Download failed for $f." >&2
    echo "EmbeddingGemma is gated: accept the license at https://huggingface.co/google/embeddinggemma-300m" >&2
    echo "Then: huggingface-cli login  (or export HF_TOKEN=…)" >&2
    exit 1
  fi
}

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         tokenizer.model special_tokens_map.json added_tokens.json modules.json \
         config_sentence_transformers.json sentence_bert_config.json \
         1_Pooling/config.json 2_Dense/config.json 3_Dense/config.json; do
  download "$f"
done

echo "Downloading 2_Dense/model.safetensors and 3_Dense/model.safetensors ..."
download 2_Dense/model.safetensors
download 3_Dense/model.safetensors

echo "Downloading model.safetensors (~1.2GB) ..."
download model.safetensors

echo "Installed to $DEST"
ls -lh
du -sh .
