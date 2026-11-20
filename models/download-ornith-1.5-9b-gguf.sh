#!/usr/bin/env bash
# Download the smallest Ornith-1.5-9B GGUF into this models/ directory.
# AtomicChat/Ornith-1.5-9B-GGUF, MIT, not gated. general.architecture=qwen35.
# AD-IQ2_XXS-IQ1_M is the smallest file (~2.8GB). The card measures about 54%
# top-1 agreement with BF16; a higher quant is in the same repository.
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
FILE="Ornith-1.5-9B-AD-IQ2_XXS-IQ1_M.gguf"
DEST="$MODELS_ROOT/$FILE"
BASE="https://huggingface.co/AtomicChat/Ornith-1.5-9B-GGUF/resolve/main"

echo "Downloading $FILE (~2.8GB) into $MODELS_ROOT ..."
curl_resume "$DEST" "$BASE/$FILE"

echo "Installed to $DEST"
ls -lh "$DEST"
du -sh "$DEST"
echo "This file is general.architecture=qwen35. Load it with LlmModelFactory.make as Qwen3.5 text chat."
