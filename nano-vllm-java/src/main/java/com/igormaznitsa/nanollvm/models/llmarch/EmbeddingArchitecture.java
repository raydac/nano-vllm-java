package com.igormaznitsa.nanollvm.models.llmarch;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;

/**
 * Template for embedding families: {@link #createEmbedding} required, never a causal chat graph.
 *
 * @since 1.1.0
 */
abstract sealed class EmbeddingArchitecture implements ArchitectureProcessor permits BertProcessor {

  /**
   * Embedding families are never causal chat graphs.
   *
   * @return {@code true}
   * @since 1.1.0
   */
  @Override
  public final boolean isEmbedding() {
    return true;
  }

  /**
   * Immutable sentence encoder for this family.
   *
   * @param config  bound Hugging Face / GGUF-derived config
   * @param weights filled parameter bag
   * @return family encoder
   * @since 1.1.0
   */
  @Override
  public abstract EmbeddingEncoder createEmbedding(final Config.HfConfig config,
                                                   final WeightBag weights);
}
