#!/usr/bin/env bash
# Download the seven YOLO26 nano ONNX files (detect, seg, sem, depth, cls, pose, obb).
# Image models, not chat. Source: https://github.com/ultralytics/assets/releases/tag/v8.4.0
set -euo pipefail

MODELS_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_curl_resume.sh
. "$MODELS_ROOT/_curl_resume.sh"
DEST="$MODELS_ROOT/YOLO26n"
BASE="https://github.com/ultralytics/assets/releases/download/v8.4.0"

download() {
  local folder="$1"
  local file="$2"
  mkdir -p "$DEST/$folder"
  echo "Downloading $file ..."
  if ! curl_resume "$DEST/$folder/$file" "$BASE/$file"; then
    echo "Download failed for $file." >&2
    echo "Release: https://github.com/ultralytics/assets/releases/tag/v8.4.0" >&2
    exit 1
  fi
}

download detect yolo26n.onnx
download seg yolo26n-seg.onnx
download sem yolo26n-sem.onnx
download depth yolo26n-depth.onnx
download cls yolo26n-cls.onnx
download pose yolo26n-pose.onnx
download obb yolo26n-obb.onnx

echo "Installed to $DEST"
find "$DEST" -type f -name '*.onnx' -exec ls -lh {} +
echo "Class names are classes.txt in the same folder as the ONNX file. Depth has none."
echo "Replace classes.txt when the head was trained on another list."
echo "Load: LlmModelFactory.make(Path.of(\"models/YOLO26n/detect/yolo26n.onnx\"))"
