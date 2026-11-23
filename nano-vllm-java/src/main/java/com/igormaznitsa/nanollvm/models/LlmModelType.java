package com.igormaznitsa.nanollvm.models;

/**
 * Kind passed to {@link LlmModelFactory#open(java.nio.file.Path, LlmModelType)}.
 * {@link LlmModelFactory#open(java.nio.file.Path)} detects one of these and calls that method.
 * GGUF and Hugging Face still read the architecture from the file. A YOLO constant selects the
 * head; the scale is still read from the weights.
 *
 * @since 1.5.0
 */
public enum LlmModelType {
  GGUF,
  HUGGING_FACE,
  PIPER,
  /**
   * Tiny Voca text-to-speech folder ({@code config.json} plus safetensors, GGUF, or ONNX)
   * or a tiny-voca {@code .gguf} file.
   *
   * @since 1.5.1
   */
  TINY_VOCA,
  FASTTEXT,
  YOLO_DETECT,
  YOLO_SEGMENT,
  YOLO_SEMANTIC,
  YOLO_DEPTH,
  YOLO_CLASSIFY,
  YOLO_POSE,
  YOLO_OBB
}
