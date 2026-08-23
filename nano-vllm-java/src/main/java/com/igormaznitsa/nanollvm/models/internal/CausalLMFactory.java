package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.llmarch.ArchitectureProcessors;

/**
 * Builds an immutable {@link CausalLM} from HF config + {@link WeightBag}.
 * Architecture is resolved by {@link ModelSupport} (optional {@code -Dnanollvm.arch=…} only
 * when it matches the checkpoint); graph construction is the matching {@code ArchitectureProcessor}.
 */
public final class CausalLMFactory {

  private CausalLMFactory() {
  }

  public static CausalLM create(final Config.HfConfig config, final WeightBag weights) {
    return ArchitectureProcessors.of(detect(config)).createCausal(config, weights);
  }

  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  public static String detect(final Config.HfConfig config) {
    ModelSupport.Selection selected = ModelSupport.resolve(config);
    if (selected.isSpeech()) {
      throw new IllegalArgumentException(
        ModelSupport.speechEngineMisuseMessage(selected.architectureId()));
    }
    if (selected.isEmbedding()) {
      throw new IllegalArgumentException(ModelSupport.chatMisuseMessage(selected.architectureId()));
    }
    return selected.architectureId();
  }
}
