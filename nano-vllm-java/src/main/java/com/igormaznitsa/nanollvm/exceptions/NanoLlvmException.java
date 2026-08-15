package com.igormaznitsa.nanollvm.exceptions;

/**
 * Base type for library runtime failures thrown by load, generate, and related entry points.
 * Argument validation may still throw {@link IllegalArgumentException}.
 */
public class NanoLlvmException extends RuntimeException {

  /**
   * @param message user-facing failure text
   */
  public NanoLlvmException(final String message) {
    super(message);
  }

  /**
   * @param message user-facing failure text
   * @param cause   underlying failure
   */
  public NanoLlvmException(final String message, final Throwable cause) {
    super(message, cause);
  }

  protected NanoLlvmException() {
    super();
  }
}
