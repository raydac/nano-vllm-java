#!/usr/bin/env bash
# Download FacebookAI/xlm-roberta-base as an ONNX embedding folder.
# XLM-RoBERTa encoder (fill-mask checkpoint, mean-pooled embeddings) — not a causal chat model.
# Source: https://huggingface.co/FacebookAI/xlm-roberta-base
# Fetches model.onnx (~1.9GB fp32) into onnx/model.onnx so the loader sees the expected name.
# Do not copy model.safetensors — safetensors would win and HF BERT-family safetensors is not loaded.
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$MODELS_ROOT/xlm-roberta-base"
BASE="https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main"

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
elif [[ -n "${HF_HOME:-}" && -f "${HF_HOME}/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HF_HOME}/token")")
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  AUTH=(-H "Authorization: Bearer $(cat "${HOME}/.cache/huggingface/token")")
fi

mkdir -p "$DEST/onnx"
cd "$DEST"

download() {
  local dest="$1"
  local src="${2:-$1}"
  echo "Downloading $src -> $dest ..."
  if ! curl -L --fail --retry 3 -C - "${AUTH[@]}" -o "$dest" "$BASE/$src"; then
    echo "Download failed for $src." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Model card: https://huggingface.co/FacebookAI/xlm-roberta-base" >&2
    exit 1
  fi
}

echo "Downloading config / tokenizer sidecars ..."
for f in config.json tokenizer.json tokenizer_config.json; do
  download "$f"
done

echo "Downloading onnx/model.onnx (~1.9GB fp32) from Hugging Face model.onnx ..."
download onnx/model.onnx model.onnx

echo "Installed to $DEST"
ls -lh
ls -lh onnx
du -sh .
echo "Note: embedding encoder (XLM-RoBERTa / BERT graph), context up to 512 tokens — not for LLM chat samples."
echo "Load: LlmModelFactory.make(Path.of(\"models/xlm-roberta-base\")); model.embed(text)"
