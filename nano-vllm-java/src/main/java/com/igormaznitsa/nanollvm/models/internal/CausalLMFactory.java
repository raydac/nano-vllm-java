package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;

/**
 * Builds an immutable {@link CausalLM} from HF config + {@link WeightBag}.
 * Architecture is resolved by {@link ModelSupport} (optional {@code -Dnanollvm.arch=…} only
 * when it matches the checkpoint).
 */
public final class CausalLMFactory {

  private CausalLMFactory() {
  }

  public static CausalLM create(final Config.HfConfig config, final WeightBag weights) {
    return switch (detect(config)) {
      case ARCH_GEMMA3 -> new Gemma3ForCausalLM(config, weights);
      case ARCH_QWEN3 -> new Qwen3ForCausalLM(config, weights);
      case ARCH_LLAMA -> new LlamaForCausalLM(config, weights);
      case ARCH_LFM2 -> new Lfm2ForCausalLM(config, weights);
      default -> throw new IllegalStateException("unsupported chat architecture after detect");
    };
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  public static String detect(final Config.HfConfig config) {
    ModelSupport.Selection selected = ModelSupport.resolve(config);
    if (selected.isEmbedding()) {
      throw new IllegalArgumentException(ModelSupport.chatMisuseMessage(selected.architectureId()));
    }
    return selected.architectureId();
  }
}
