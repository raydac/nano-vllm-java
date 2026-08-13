package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;

/**
 * Builds an immutable {@link EmbeddingEncoder} from HF/GGUF config + {@link WeightBag}.
 *
 * @since 1.1.0
 */
public final class EmbeddingEncoderFactory {

  private EmbeddingEncoderFactory() {
  }

  public static EmbeddingEncoder create(final Config.HfConfig config, final WeightBag weights) {
    return switch (detect(config)) {
      case ARCH_BERT -> new BertForEmbedding(config, weights);
      default -> throw new IllegalStateException("unsupported embedding architecture after detect");
    };
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  public static boolean isEmbeddingArchitecture(final Config.HfConfig config) {
    return ModelSupport.isEmbedding(config);
  }

  public static String detect(final Config.HfConfig config) {
    ModelSupport.Selection selected = ModelSupport.resolve(config);
    if (!selected.isEmbedding()) {
      throw new IllegalArgumentException(
        ModelSupport.embedMisuseMessage(selected.architectureId()));
    }
    return selected.architectureId();
  }
}
