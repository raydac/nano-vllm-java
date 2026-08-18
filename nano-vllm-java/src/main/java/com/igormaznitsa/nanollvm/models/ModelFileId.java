package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Logical Hugging Face / GGUF / ONNX file roles for {@link ModelFileSource}.
 *
 * <p>{@link #fileName()} is the conventional on-disk / resource name adapters should use.
 * {@link #MODEL_ONNX} and {@link #MODEL_ONNX_FP16} are Tier A weight imports (<strong>since
 * 1.1.0</strong>); adapters may also resolve the same names under an {@code onnx/} subfolder.
 *
 * @since 1.1.0
 */
public enum ModelFileId {

  CONFIG("config.json"),
  TOKENIZER("tokenizer.json"),
  TOKENIZER_CONFIG("tokenizer_config.json"),
  /**
   * SentencePiece protobuf vocab used when {@link #TOKENIZER} is absent.
   *
   * @since 1.1.1
   */
  TOKENIZER_MODEL("tokenizer.model"),
  GENERATION_CONFIG("generation_config.json"),
  ADDED_TOKENS("added_tokens.json"),
  SPECIAL_TOKENS_MAP("special_tokens_map.json"),
  SAFE_TENSORS_INDEX("model.safetensors.index.json"),
  MODEL_SAFE_TENSORS("model.safetensors"),
  /**
   * FP32 / default ONNX weight file (root or {@code onnx/}). @since 1.1.0
   */
  MODEL_ONNX("model.onnx"),
  /** FP16 ONNX weight file (root or {@code onnx/}). @since 1.1.0 */
  MODEL_ONNX_FP16("model_fp16.onnx"),
  GGUF("model.gguf");

  private final String fileName;

  ModelFileId(final String fileName) {
    this.fileName = requireNonNull(fileName, "fileName");
  }

  /**
   * Conventional on-disk / resource file name for this role.
   *
   * @since 1.1.0
   */
  public String fileName() {
    return this.fileName;
  }
}
