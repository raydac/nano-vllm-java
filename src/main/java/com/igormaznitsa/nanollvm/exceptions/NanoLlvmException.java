package com.igormaznitsa.nanollvm.exceptions;

public class NanoLlvmException extends RuntimeException {

  public NanoLlvmException(String message) {
    super(message);
  }

  public NanoLlvmException(String message, Throwable cause) {
    super(message, cause);
  }

  protected NanoLlvmException() {
    super();
  }
}
