#!/usr/bin/env bash
# Download Meta fastText lid.176.bin (language identification, 176 languages).
# Denser than lid.176.ftz — slightly more accurate (and faster) per Meta.
# Text classification (text -> labels). Not a chat / embedding / Whisper / Piper model.
# Docs: https://fasttext.cc/docs/en/language-identification.html
# Model: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/fasttext-lid-176"
URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FILE="lid.176.bin"

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading $FILE (~126MB) ..."
if ! curl_resume "$FILE" "$URL"; then
  echo "Download failed for $FILE." >&2
  echo "Source: https://fasttext.cc/docs/en/language-identification.html" >&2
  exit 1
fi

echo "Installed to $DEST"
ls -lh
du -sh .
echo "Load: LlmModelFactory.make(Path.of(\"models/fasttext-lid-176\"));"
echo "      llm.generate(LlmInText.of(\"Bonjour\"), LlmModality.LABELS);"
echo "Try: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LanguageIdHelloWorld"
