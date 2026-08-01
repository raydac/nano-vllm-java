package com.igormaznitsa.nanollvm;

import java.time.Duration;

/**
 * Thrown when a {@link LLM#generate} call exceeds its timeout.
 */
public final class GenerationTimeoutException extends RuntimeException {

  private final Duration timeout;

  public GenerationTimeoutException(Duration timeout) {
    super("generation timed out after " + timeout);
    this.timeout = timeout;
  }

  public Duration timeout() {
    return this.timeout;
  }
}
