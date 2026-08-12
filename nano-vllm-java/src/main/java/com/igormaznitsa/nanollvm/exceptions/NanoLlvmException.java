package com.igormaznitsa.nanollvm.exceptions;

/**
 * Base type for library runtime failures thrown by load, generate, and related entry points.
 * Argument validation may still throw {@link IllegalArgumentException}.
 */
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
