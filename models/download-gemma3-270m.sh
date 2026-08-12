#!/usr/bin/env bash
# Download Gemma 3 270M IT weights into models/Gemma3-270M.
# Requires accepting the Gemma license on Hugging Face and usually HF_TOKEN.
#   huggingface-cli login
#   export HF_TOKEN=hf_...
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/Gemma3-270M"
BASE="https://huggingface.co/google/gemma-3-270m-it/resolve/main"

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
  if ! curl -L --fail --retry 3 -C - "${AUTH[@]}" -o "$f" "$BASE/$f"; then
    echo "Download failed for $f." >&2
    echo "Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-270m-it" >&2
    echo "Then: huggingface-cli login  (or export HF_TOKEN=…)" >&2
    exit 1
  fi
}

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         tokenizer.model special_tokens_map.json added_tokens.json; do
  download "$f"
done

echo "Downloading model.safetensors (~0.5GB) ..."
download model.safetensors

echo "Installed to $DEST"
ls -lh
du -sh .
