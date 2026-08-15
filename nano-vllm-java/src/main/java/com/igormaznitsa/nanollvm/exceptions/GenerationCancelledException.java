package com.igormaznitsa.nanollvm.exceptions;

/**
 * Thrown when {@link com.igormaznitsa.nanollvm.llm.LLM#cancel()} aborts an in-flight
 * {@link com.igormaznitsa.nanollvm.llm.LLM#generate}.
 */
public final class GenerationCancelledException extends NanoLlvmException {

  /**
   * Cancels the in-flight generate with a fixed message.
   */
  public GenerationCancelledException() {
    super("generation cancelled");
  }
}
