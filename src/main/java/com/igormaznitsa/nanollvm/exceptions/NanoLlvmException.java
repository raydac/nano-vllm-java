package com.igormaznitsa.nanollvm.exceptions;

public class NanoLlvmException extends RuntimeException {

  public NanoLlvmException(final String message) {
    super(message);
  }

  public NanoLlvmException(final String message, final Throwable cause) {
    super(message, cause);
  }

  protected NanoLlvmException() {
    super();
  }
}
