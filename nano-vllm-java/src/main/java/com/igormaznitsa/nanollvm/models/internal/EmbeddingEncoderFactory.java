package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.llmarch.ArchitectureProcessors;

/**
 * Builds an immutable {@link EmbeddingEncoder} from HF/GGUF config + {@link WeightBag}.
 *
 * @since 1.1.0
 */
public final class EmbeddingEncoderFactory {

  private EmbeddingEncoderFactory() {
  }

  /**
   * Builds the BERT-family encoder for {@code config}.
   *
   * @throws IllegalArgumentException if {@code config} is not an embedding checkpoint
   */
  public static EmbeddingEncoder create(final Config.HfConfig config, final WeightBag weights) {
    return ArchitectureProcessors.of(detect(config)).createEmbedding(config, weights);
  }

  /**
   * Weight-name schema for the embedding architecture in {@code config}.
   */
  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  /**
   * {@code true} when {@code config} is a BERT-encoder embedding family.
   */
  public static boolean isEmbeddingArchitecture(final Config.HfConfig config) {
    return ModelSupport.isEmbedding(config);
  }

  /**
   * Architecture id for an embedding checkpoint ({@code bert} / {@code roberta} /
   * {@code xlm-roberta}).
   *
   * @throws IllegalArgumentException if {@code config} is not a supported embedding family
   */
  public static String detect(final Config.HfConfig config) {
    ModelSupport.Selection selected = ModelSupport.resolve(config);
    if (selected.isSpeech()) {
      throw new IllegalArgumentException(
        ModelSupport.speechEmbedMisuseMessage(selected.architectureId()));
    }
    if (selected.isSynthesis()) {
      throw new IllegalArgumentException(
        ModelSupport.synthesisEmbedMisuseMessage(selected.architectureId()));
    }
    if (!selected.isEmbedding()) {
      throw new IllegalArgumentException(
        ModelSupport.embedMisuseMessage(selected.architectureId()));
    }
    return selected.architectureId();
  }
}
