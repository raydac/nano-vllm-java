package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Already-tokenized ids for embedding via {@link LlmModel#generate(LlmInput, LlmModality)}.
 * Include special tokens such as {@code [CLS]} / {@code [SEP]} (or {@code <s>} / {@code </s>})
 * when the encoder requires them.
 *
 * @param tokenIds non-empty token ids; never {@code null}
 * @since 1.3.0
 */
@SuppressWarnings("ArrayRecordComponent")
public record LlmInTokenIds(int[] tokenIds) implements LlmInput {

  /**
   * @throws NullPointerException     if {@code tokenIds} is {@code null}
   * @throws IllegalArgumentException if {@code tokenIds} is empty
   */
  public LlmInTokenIds {
    requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.length == 0) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    tokenIds = tokenIds.clone();
  }

  /**
   * Creates a token-id embedding input.
   *
   * @param tokenIds non-empty ids; must not be {@code null}
   * @return input wrapping a defensive copy
   */
  public static LlmInTokenIds of(final int[] tokenIds) {
    return new LlmInTokenIds(tokenIds);
  }

  @Override
  public int[] tokenIds() {
    return this.tokenIds.clone();
  }
}
