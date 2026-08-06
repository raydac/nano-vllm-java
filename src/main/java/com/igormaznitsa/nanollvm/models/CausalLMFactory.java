package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_ARCH;
import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.llm.Config;

import java.util.List;

/**
 * Builds an immutable {@link CausalLM} from HF config + {@link WeightBag}
 * (optional {@code -Dnanovllm.arch=qwen3|gemma3|lfm2}).
 */
public final class CausalLMFactory {

  private CausalLMFactory() {
  }

  public static CausalLM create(final Config.HfConfig config, final WeightBag weights) {
    String forced = System.getProperty(PROP_ARCH, "").strip().toLowerCase(ROOT);
    String arch = forced.isEmpty() ? detect(config) : normalize(forced);
    return switch (arch) {
      case ARCH_GEMMA3 -> new Gemma3ForCausalLM(config, weights);
      case ARCH_QWEN3 -> new Qwen3ForCausalLM(config, weights);
      case ARCH_LFM2 -> new Lfm2ForCausalLM(config, weights);
      default -> throw new IllegalArgumentException(
        "unsupported architecture '" + arch + "' (use qwen3|gemma3|lfm2; set -D" + PROP_ARCH + "=…)"
      );
    };
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    String forced = System.getProperty(PROP_ARCH, "").strip().toLowerCase(ROOT);
    String arch = forced.isEmpty() ? detect(config) : normalize(forced);
    return WeightSchema.forArchitecture(arch, config);
  }

  public static String detect(final Config.HfConfig config) {
    if (config.modelType() != null) {
      String mt = config.modelType().toLowerCase(ROOT);
      if (mt.contains("gemma")) {
        return ARCH_GEMMA3;
      }
      if (mt.contains("lfm2")) {
        return ARCH_LFM2;
      }
      if (mt.contains("qwen")) {
        return ARCH_QWEN3;
      }
    }
    List<String> architectures = config.architectures();
    if (architectures != null) {
      for (String a : architectures) {
        if (a == null) {
          continue;
        }
        String lower = a.toLowerCase(ROOT);
        if (lower.contains("gemma")) {
          return ARCH_GEMMA3;
        }
        if (lower.contains("lfm2")) {
          return ARCH_LFM2;
        }
        if (lower.contains("qwen")) {
          return ARCH_QWEN3;
        }
      }
    }
    return ARCH_QWEN3;
  }

  private static String normalize(final String arch) {
    return switch (arch) {
      case "gemma", ARCH_GEMMA3, "gemma3_text" -> ARCH_GEMMA3;
      case "qwen", ARCH_QWEN3 -> ARCH_QWEN3;
      case "lfm", ARCH_LFM2, "lfm2.5" -> ARCH_LFM2;
      default -> arch;
    };
  }
}
