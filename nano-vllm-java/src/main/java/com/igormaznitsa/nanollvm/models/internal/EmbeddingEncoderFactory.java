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

  public static EmbeddingEncoder create(final Config.HfConfig config, final WeightBag weights) {
    return ArchitectureProcessors.of(detect(config)).createEmbedding(config, weights);
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  public static boolean isEmbeddingArchitecture(final Config.HfConfig config) {
    return ModelSupport.isEmbedding(config);
  }

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
