#!/usr/bin/env bash
# Download onnx-community SmolLM2-135M-Instruct-ONNX (Llama ChatML, ~135M).
# Prefer fp16 ONNX (~270 MiB) — loader converts to float32.
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/SmolLM2-135M-Instruct-ONNX"
BASE="https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/resolve/main"

mkdir -p "$DEST/onnx"
cd "$DEST"

echo "Downloading config / generation_config ..."
curl -L --fail --retry 3 -C - -o config.json "$BASE/config.json"
curl -L --fail --retry 3 -C - -o generation_config.json "$BASE/generation_config.json"

echo "Downloading tokenizer sidecars ..."
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  curl -L --fail --retry 3 -C - -o "$f" "$BASE/$f"
done

echo "Downloading onnx/model_fp16.onnx (~270 MiB) ..."
curl -L --fail --retry 3 -C - -o onnx/model_fp16.onnx "$BASE/onnx/model_fp16.onnx"

echo "Installed to $DEST"
ls -lh
ls -lh onnx
du -sh .
