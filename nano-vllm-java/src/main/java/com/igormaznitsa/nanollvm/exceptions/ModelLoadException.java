package com.igormaznitsa.nanollvm.exceptions;

/**
 * Failure while loading a model, tokenizer, or RAG corpus (missing files, empty corpus, corrupt
 * weights, …).
 */
public class ModelLoadException extends NanoLlvmException {

  /**
   * @param message what failed to load
   */
  public ModelLoadException(final String message) {
    super(message);
  }

  /**
   * @param message what failed to load
   * @param cause   underlying I/O or parse failure
   */
  public ModelLoadException(final String message, final Throwable cause) {
    super(message, cause);
  }
}
