#!/usr/bin/env bash
# Download Qwen3-0.6B Q8_0 GGUF (general.architecture=qwen3) into this models/ directory.
# Apache 2.0, not gated. The safetensors folder is models/download-qwen3-0.6b.sh.
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
FILE="Qwen3-0.6B-Q8_0.gguf"
DEST="$MODELS_ROOT/$FILE"
BASE="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main"

echo "Downloading $FILE (~570MB) into $MODELS_ROOT ..."
curl_resume "$DEST" "$BASE/$FILE"

echo "Installed to $DEST"
ls -lh "$DEST"
du -sh "$DEST"
echo "Hint: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Example -Dexec.args=$DEST"
