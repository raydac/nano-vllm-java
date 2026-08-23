#!/usr/bin/env bash
# Download ChristianAzinn gte-small Q2_K GGUF (smallest quant, ~25MB) into this models/ directory.
# Embedding / feature-extraction BERT GGUF — not a causal chat model for Example/HelloWorld.
# Source: https://huggingface.co/ChristianAzinn/gte-small-gguf
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
FILE="gte-small.Q2_K.gguf"
DEST="$MODELS_ROOT/$FILE"
BASE="https://huggingface.co/ChristianAzinn/gte-small-gguf/resolve/main"

echo "Downloading $FILE (~25MB, smallest GTE-small GGUF) into $MODELS_ROOT ..."
curl_resume "$DEST" "$BASE/$FILE"

echo "Installed to $DEST"
ls -lh "$DEST"
du -sh "$DEST"
echo "Note: embedding model (BERT), context up to 512 tokens — not for LLM chat samples."
