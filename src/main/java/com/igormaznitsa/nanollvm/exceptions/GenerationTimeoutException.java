package com.igormaznitsa.nanollvm.exceptions;

import java.time.Duration;

public final class GenerationTimeoutException extends NanoLlvmException {

  private final Duration timeout;

  public GenerationTimeoutException(Duration timeout) {
    super("generation timed out after " + timeout);
    this.timeout = timeout;
  }

  public Duration timeout() {
    return this.timeout;
  }
}
