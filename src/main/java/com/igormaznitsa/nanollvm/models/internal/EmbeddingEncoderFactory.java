package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.llm.Config;

/**
 * Builds an immutable {@link EmbeddingEncoder} from HF/GGUF config + {@link WeightBag}.
 *
 * @since 1.1.0
 */
public final class EmbeddingEncoderFactory {

  private EmbeddingEncoderFactory() {
  }

  public static EmbeddingEncoder create(final Config.HfConfig config, final WeightBag weights) {
    String arch = detect(config);
    return switch (arch) {
      case ARCH_BERT -> new BertForEmbedding(config, weights);
      default -> throw new IllegalArgumentException(
        "unsupported embedding architecture '" + arch + "' (expected bert)");
    };
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  public static boolean isEmbeddingArchitecture(final Config.HfConfig config) {
    return ARCH_BERT.equals(detect(config));
  }

  public static String detect(final Config.HfConfig config) {
    if (config.modelType() != null) {
      String mt = config.modelType().toLowerCase(ROOT);
      if (mt.contains("bert")) {
        return ARCH_BERT;
      }
    }
    if (config.architectures() != null) {
      for (String a : config.architectures()) {
        if (a != null && a.toLowerCase(ROOT).contains("bert")) {
          return ARCH_BERT;
        }
      }
    }
    throw new IllegalArgumentException("not an embedding architecture: " + config.modelType());
  }
}
