#!/usr/bin/env bash
# Download LiquidAI LFM2.5-2.6B Q4_K_M GGUF into this models/ directory.
# Dequantizes to ~10GB float32 at load — plan on -Xmx16g (default in .mvn/jvm.config).
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE="LFM2.5-2.6B-Q4_K_M.gguf"
DEST="$MODELS_ROOT/$FILE"
BASE="https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main"

echo "Downloading $FILE (~1.67GB) into $MODELS_ROOT ..."
curl -L --fail --retry 3 -C - -o "$DEST" "$BASE/$FILE"

echo "Installed to $DEST"
ls -lh "$DEST"
du -sh "$DEST"
echo "Hint: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Example -Dexec.args=$DEST"
