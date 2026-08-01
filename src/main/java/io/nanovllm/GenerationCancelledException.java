package io.nanovllm;

/**
 * Thrown when {@link LLM#cancel()} stops an in-flight {@link LLM#generate} call.
 */
public final class GenerationCancelledException extends RuntimeException {

  public GenerationCancelledException() {
    super("generation cancelled");
  }
}
