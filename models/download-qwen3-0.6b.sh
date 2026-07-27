#!/usr/bin/env bash
# Download Qwen3-0.6B weights into this models/ directory (not into the git tree of src/).
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/Qwen3-0.6B"
BASE="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"

mkdir -p "$DEST"
cd "$DEST"

for f in config.json generation_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json; do
  echo "Downloading $f ..."
  curl -L --fail --retry 3 -C - -o "$f" "$BASE/$f"
done

echo "Downloading model.safetensors (~1.4GB) ..."
curl -L --fail --retry 3 -C - -o model.safetensors "$BASE/model.safetensors"

echo "Installed to $DEST"
ls -lh
du -sh .
