package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_ARCH;
import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;

import java.util.List;
import java.util.Optional;

/**
 * Builds an immutable {@link CausalLM} from HF config + {@link WeightBag}
 * (optional {@code -Dnanollvm.arch=qwen3|gemma3|llama|lfm2}). Llama support is
 * <strong>since 1.1.0</strong> ({@link LlamaForCausalLM}).
 */
public final class CausalLMFactory {

  private CausalLMFactory() {
  }

  public static CausalLM create(final Config.HfConfig config, final WeightBag weights) {
    String arch = resolveArch(config);
    return switch (arch) {
      case ARCH_GEMMA3 -> new Gemma3ForCausalLM(config, weights);
      case ARCH_QWEN3 -> new Qwen3ForCausalLM(config, weights);
      case ARCH_LLAMA -> new LlamaForCausalLM(config, weights);
      case ARCH_LFM2 -> new Lfm2ForCausalLM(config, weights);
      default -> throw new IllegalArgumentException(
        "unsupported architecture '" + arch
          + "' (use qwen3|gemma3|llama|lfm2; set -D" + PROP_ARCH + "=…)"
      );
    };
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(resolveArch(config), config);
  }

  private static String resolveArch(final Config.HfConfig config) {
    String forced = Optional.ofNullable(NanoLlvmProps.systemProperty(PROP_ARCH))
      .orElse("")
      .strip()
      .toLowerCase(ROOT);
    return forced.isEmpty() ? detect(config) : normalize(forced);
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
      if (mt.contains("llama")) {
        return ARCH_LLAMA;
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
        if (lower.contains("llama")) {
          return ARCH_LLAMA;
        }
        if (lower.contains("qwen")) {
          return ARCH_QWEN3;
        }
      }
    }
    throw new IllegalArgumentException(
      "cannot detect architecture from config (model_type/architectures); set -D"
        + PROP_ARCH + "=qwen3|gemma3|llama|lfm2"
    );
  }

  private static String normalize(final String arch) {
    return switch (arch) {
      case "gemma", ARCH_GEMMA3, "gemma3_text" -> ARCH_GEMMA3;
      case "qwen", ARCH_QWEN3 -> ARCH_QWEN3;
      case ARCH_LLAMA, "llama2", "llama3" -> ARCH_LLAMA;
      case "lfm", ARCH_LFM2, "lfm2.5" -> ARCH_LFM2;
      default -> arch;
    };
  }
}
