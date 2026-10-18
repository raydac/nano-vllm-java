#!/usr/bin/env bash
# Download Jeff-Qwen3.5-0.8B (text) into models/Jeff-Qwen3.5-0.8B.
# Apache 2.0, ungated. Vision files and the Jeff readout head are not fetched.
# Optional HF_TOKEN if Hugging Face rate-limits you.
#   https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/Jeff-Qwen3.5-0.8B"
BASE="https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B/resolve/main"

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
  if ! curl_resume "$f" "$BASE/$f" "${AUTH[@]}"; then
    echo "Download failed for $f." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Model card: https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B" >&2
    exit 1
  fi
}

for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  download "$f"
done

echo "Downloading model.safetensors (~1.7GB) ..."
download model.safetensors

echo "Installed to $DEST"
ls -lh
du -sh .
