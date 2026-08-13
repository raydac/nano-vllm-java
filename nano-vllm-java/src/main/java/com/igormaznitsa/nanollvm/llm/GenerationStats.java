package com.igormaznitsa.nanollvm.llm;

/**
 * Engine-measured token counts and wall time for one sequence completed by {@link LLM#generate}.
 *
 * <p>{@link #elapsedNanos()} is the wall time of the enclosing {@code generate} call. In a
 * multi-prompt batch every output shares the same elapsed time (the batch ran together). Chat
 * turns expose the same object on {@link com.igormaznitsa.nanollvm.chat.ChatReply#stats()}.
 * {@link #NONE} means “not measured yet” (streaming snapshots, {@code ChatReply.parse} without
 * {@code withStats}).
 *
 * <pre>{@code
 * GenerationStats stats = output.stats();
 * int newTokens = stats.completionTokens();
 * double tokPerSec = stats.completionTokensPerSecond();
 * }</pre>
 *
 * @param promptTokens     prompt length in tokens (prefill input for this sequence)
 * @param completionTokens newly generated tokens (matches {@link LLM.GenerationOutput#tokenIds()}
 *                         size)
 * @param elapsedNanos     wall-clock duration of the generate call that produced this output
 */
public record GenerationStats(int promptTokens, int completionTokens, long elapsedNanos) {

  /**
   * Placeholder when generate has not finished or stats were never attached (all zeros).
   */
  public static final GenerationStats NONE = new GenerationStats(0, 0, 0);

  public GenerationStats {
    if (promptTokens < 0 || completionTokens < 0 || elapsedNanos < 0) {
      throw new IllegalArgumentException("stats fields must be non-negative");
    }
  }

  /**
   * {@link #promptTokens()} + {@link #completionTokens()}.
   */
  public int totalTokens() {
    return this.promptTokens + this.completionTokens;
  }

  /**
   * Completion throughput for this output using {@link #elapsedNanos} as the denominator.
   *
   * @return tokens per second, or {@code 0} when elapsed or completion count is zero
   */
  public double completionTokensPerSecond() {
    if (this.elapsedNanos == 0L || this.completionTokens == 0) {
      return 0d;
    }
    return this.completionTokens / (this.elapsedNanos / 1e9d);
  }
}
