#!/usr/bin/env bash
# Download onnx-community Tiny-LLM-ONNX (Llama, ~10M) + tokenizer from arnir0/Tiny-LLM.
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/Tiny-LLM-ONNX"
ONNX_BASE="https://huggingface.co/onnx-community/Tiny-LLM-ONNX/resolve/main"
TOK_BASE="https://huggingface.co/arnir0/Tiny-LLM/resolve/main"

mkdir -p "$DEST/onnx"
cd "$DEST"

echo "Downloading config / generation_config ..."
curl -L --fail --retry 3 -C - -o config.json "$ONNX_BASE/config.json"
curl -L --fail --retry 3 -C - -o generation_config.json "$ONNX_BASE/generation_config.json"

echo "Downloading tokenizer from arnir0/Tiny-LLM ..."
for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
  curl -L --fail --retry 3 -C - -o "$f" "$TOK_BASE/$f" || true
done

echo "Downloading onnx/model.onnx (fp32) ..."
curl -L --fail --retry 3 -C - -o onnx/model.onnx "$ONNX_BASE/onnx/model.onnx"

echo "Installed to $DEST"
ls -lh
ls -lh onnx
du -sh .
