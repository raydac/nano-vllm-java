#!/usr/bin/env bash
# Download Piper US English Lessac medium (ONNX + sidecar) and espeak-ng-data.
# Text-to-speech (text -> WAV). Not a chat model. Not ONNX Runtime.
# Voice: https://huggingface.co/rhasspy/piper-voices
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
# shellcheck source=./_espeak_ng_data.sh
. "$MODELS_ROOT/_espeak_ng_data.sh"
DEST="$MODELS_ROOT/piper-en-lessac-medium"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

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
  if ! curl_resume "$f" "$VOICE_BASE/$f" "${AUTH[@]}"; then
    echo "Download failed for $f." >&2
    echo "Retry, or export HF_TOKEN=… / huggingface-cli login if Hugging Face rate-limits you." >&2
    echo "Voice card: https://huggingface.co/rhasspy/piper-voices" >&2
    exit 1
  fi
}

echo "Downloading Piper Lessac medium ONNX + sidecar ..."
download en_US-lessac-medium.onnx
download en_US-lessac-medium.onnx.json

install_espeak_ng_data "$DEST"

echo "Installed to $DEST"
ls -lh
du -sh .
echo "Load: LlmModelFactory.open(Path.of(\"models/piper-en-lessac-medium\"))"
echo "      .optionalData(LlmOptionalData.ESPEAK_DATA, Path.of(\"models/piper-en-lessac-medium/espeak-ng-data\"))"
echo "      .make(); model.synthesize(\"Hello world\")"
echo "Try: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.SynthesizeHelloWorld -Dexec.args='models/piper-en-lessac-medium Hello world'"
