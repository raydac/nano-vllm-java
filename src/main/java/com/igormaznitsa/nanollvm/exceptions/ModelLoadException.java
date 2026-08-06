package com.igormaznitsa.nanollvm.exceptions;

public final class ModelLoadException extends NanoLlvmException {

  public ModelLoadException(final String message) {
    super(message);
  }

  public ModelLoadException(final String message, final Throwable cause) {
    super(message, cause);
  }
}
