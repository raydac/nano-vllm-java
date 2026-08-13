package com.igormaznitsa.nanollvm.exceptions;

/**
 * Failure while loading a model, tokenizer, or RAG corpus (missing files, empty corpus, corrupt
 * weights, …).
 */
public class ModelLoadException extends NanoLlvmException {

  public ModelLoadException(final String message) {
    super(message);
  }

  public ModelLoadException(final String message, final Throwable cause) {
    super(message, cause);
  }
}
