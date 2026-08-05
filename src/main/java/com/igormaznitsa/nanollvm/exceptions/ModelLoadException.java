package com.igormaznitsa.nanollvm.exceptions;

public final class ModelLoadException extends NanoLlvmException {

  public ModelLoadException(String message) {
    super(message);
  }

  public ModelLoadException(String message, Throwable cause) {
    super(message, cause);
  }
}
