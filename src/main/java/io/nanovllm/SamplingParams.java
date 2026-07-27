package io.nanovllm;

public record SamplingParams(
    float temperature,
    int maxTokens,
    boolean ignoreEos,
    int topK,
    float topP
) {

  public SamplingParams {
    if (temperature <= 1e-10f) {
      throw new IllegalArgumentException("greedy sampling is not permitted");
    }
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be >= 1");
    }
    if (topK < 0) {
      throw new IllegalArgumentException("topK must be >= 0 (0 = disabled)");
    }
    if (topP <= 0f || topP > 1f) {
      throw new IllegalArgumentException("topP must be in (0, 1]");
    }
  }

  public SamplingParams() {
    this(0.7f, 64, false, 0, 0.9f);
  }

  public SamplingParams(float temperature, int maxTokens) {
    this(temperature, maxTokens, false, 0, 0.9f);
  }

  public SamplingParams(float temperature, int maxTokens, boolean ignoreEos) {
    this(temperature, maxTokens, ignoreEos, 0, 0.9f);
  }
}
