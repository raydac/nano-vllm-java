package com.igormaznitsa.nanollvm.exceptions;

import java.time.Duration;

/**
 * Thrown when a {@link com.igormaznitsa.nanollvm.llm.LLM#generate} wall-clock timeout elapses.
 */
public final class GenerationTimeoutException extends NanoLlvmException {

  private final Duration timeout;

  /**
   * @param timeout wall-clock limit that elapsed; included in {@link #getMessage()}
   */
  public GenerationTimeoutException(final Duration timeout) {
    super("generation timed out after " + timeout);
    this.timeout = timeout;
  }

  /**
   * Deadline that elapsed.
   */
  public Duration timeout() {
    return this.timeout;
  }
}
