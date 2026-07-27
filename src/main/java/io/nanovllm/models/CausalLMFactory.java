package io.nanovllm.models;

import io.nanovllm.Config;

import java.util.List;
import java.util.Locale;

/**
 * Builds a {@link CausalLM} from HF config (+ optional {@code -Dnanovllm.arch=qwen3|gemma3}).
 */
public final class CausalLMFactory {

  private CausalLMFactory() {
  }

  public static CausalLM create(Config.HfConfig config) {
    String forced = System.getProperty("nanovllm.arch", "").strip().toLowerCase(Locale.ROOT);
    String arch = forced.isEmpty() ? detect(config) : normalize(forced);
    return switch (arch) {
      case "gemma3" -> new Gemma3ForCausalLM(config);
      case "qwen3" -> new Qwen3ForCausalLM(config);
      default -> throw new IllegalArgumentException(
          "unsupported architecture '" + arch + "' (use qwen3|gemma3; set -Dnanovllm.arch=…)"
      );
    };
  }

  public static String detect(Config.HfConfig config) {
    if (config.modelType() != null) {
      String mt = config.modelType().toLowerCase(Locale.ROOT);
      if (mt.contains("gemma")) {
        return "gemma3";
      }
      if (mt.contains("qwen")) {
        return "qwen3";
      }
    }
    List<String> architectures = config.architectures();
    if (architectures != null) {
      for (String a : architectures) {
        if (a == null) {
          continue;
        }
        String lower = a.toLowerCase(Locale.ROOT);
        if (lower.contains("gemma")) {
          return "gemma3";
        }
        if (lower.contains("qwen")) {
          return "qwen3";
        }
      }
    }
    return "qwen3";
  }

  private static String normalize(String arch) {
    return switch (arch) {
      case "gemma", "gemma3", "gemma3_text" -> "gemma3";
      case "qwen", "qwen3" -> "qwen3";
      default -> arch;
    };
  }
}
