package io.nanovllm;

/**
 * Thrown when model weights, tokenizer, or graph construction fails.
 */
public final class ModelLoadException extends RuntimeException {

  public ModelLoadException(String message) {
    super(message);
  }

  public ModelLoadException(String message, Throwable cause) {
    super(message, cause);
  }
}
