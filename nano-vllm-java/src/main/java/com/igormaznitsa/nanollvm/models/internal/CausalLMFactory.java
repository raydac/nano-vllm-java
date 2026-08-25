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

  /**
   * Builds the causal graph for {@code config}'s architecture.
   *
   * @throws IllegalArgumentException if {@code config} is an embedding, speech, or synthesis
   *                                  checkpoint
   */
  public static CausalLM create(final Config.HfConfig config, final WeightBag weights) {
    return ArchitectureProcessors.of(detect(config)).createCausal(config, weights);
  }

  /**
   * Weight-name schema for the causal architecture in {@code config}.
   */
  public static WeightSchema schema(final Config.HfConfig config) {
    return WeightSchema.forArchitecture(detect(config), config);
  }

  /**
   * Architecture id for a causal checkpoint ({@code qwen3}, {@code gemma3}, {@code gemma4},
   * {@code llama}, {@code lfm2}).
   *
   * @throws IllegalArgumentException if {@code config} is not a supported causal family
   */
  public static String detect(final Config.HfConfig config) {
    ModelSupport.Selection selected = ModelSupport.resolve(config);
    if (selected.isSpeech()) {
      throw new IllegalArgumentException(
        ModelSupport.speechEngineMisuseMessage(selected.architectureId()));
    }
    if (selected.isSynthesis()) {
      throw new IllegalArgumentException(
        ModelSupport.synthesisEngineMisuseMessage(selected.architectureId()));
    }
    if (selected.isEmbedding()) {
      throw new IllegalArgumentException(ModelSupport.chatMisuseMessage(selected.architectureId()));
    }
    return selected.architectureId();
  }
}
