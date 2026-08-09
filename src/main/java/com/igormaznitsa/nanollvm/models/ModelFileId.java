package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Logical Hugging Face / GGUF file roles for {@link ModelFileSource}.
 *
 * <p>{@link #fileName()} is the conventional on-disk / resource name adapters should use.
 *
 * @since 1.1.0
 */
public enum ModelFileId {

  CONFIG("config.json"),
  TOKENIZER("tokenizer.json"),
  TOKENIZER_CONFIG("tokenizer_config.json"),
  GENERATION_CONFIG("generation_config.json"),
  ADDED_TOKENS("added_tokens.json"),
  SPECIAL_TOKENS_MAP("special_tokens_map.json"),
  SAFE_TENSORS_INDEX("model.safetensors.index.json"),
  MODEL_SAFE_TENSORS("model.safetensors"),
  GGUF("model.gguf");

  private final String fileName;

  ModelFileId(final String fileName) {
    this.fileName = requireNonNull(fileName, "fileName");
  }

  public String fileName() {
    return this.fileName;
  }
}
