package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * L2-normalized embedding vector.
 *
 * @param vector dense floats; never {@code null} or empty
 * @since 1.3.0
 */
@SuppressWarnings("ArrayRecordComponent")
public record LlmOutEmbedding(float[] vector) implements LlmOutput {

  /**
   * @throws NullPointerException     if {@code vector} is {@code null}
   * @throws IllegalArgumentException if {@code vector} is empty
   */
  public LlmOutEmbedding {
    requireNonNull(vector, "vector");
    if (vector.length == 0) {
      throw new IllegalArgumentException("vector must not be empty");
    }
    vector = vector.clone();
  }

  @Override
  public float[] vector() {
    return this.vector.clone();
  }

  @Override
  public LlmModality modality() {
    return LlmModality.EMBEDDING;
  }
}
