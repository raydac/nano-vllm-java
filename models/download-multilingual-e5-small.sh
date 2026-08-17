#!/usr/bin/env bash
# Download intfloat/multilingual-e5-small as an ONNX embedding folder.
# BERT graph + XLM-RoBERTa Unigram tokenizer — not a causal chat model.
# Source: https://huggingface.co/intfloat/multilingual-e5-small
# HF BERT safetensors are not loaded; this script fetches onnx/model.onnx (~470MB fp32).
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/multilingual-e5-small"
BASE="https://huggingface.co/intfloat/multilingual-e5-small/resolve/main"

mkdir -p "$DEST/onnx"
cd "$DEST"

echo "Downloading config / tokenizer sidecars ..."
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -L --fail --retry 3 -C - -o "$f" "$BASE/$f"
done

echo "Downloading onnx/model.onnx (~470MB fp32) ..."
curl -L --fail --retry 3 -C - -o onnx/model.onnx "$BASE/onnx/model.onnx"

echo "Installed to $DEST"
ls -lh
ls -lh onnx
du -sh .
echo "Note: embedding model (BERT / multilingual E5), context up to 512 tokens — not for LLM chat samples."
echo "E5 expects prefixes: query: …  and  passage: …"
echo "Load: LlmModelFactory.make(Path.of(\"models/multilingual-e5-small\")); model.embed(\"query: hello\")"
