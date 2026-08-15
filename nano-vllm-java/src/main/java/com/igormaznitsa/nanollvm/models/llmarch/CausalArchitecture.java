package com.igormaznitsa.nanollvm.models.llmarch;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Template for causal chat families: Hugging Face bind by default, {@link #createCausal} required,
 * never an embedding encoder.
 *
 * <p>GGUF-only or dual-source families override {@link #bind}. Restricted fills override
 * {@link #fill}. Gemma 3 and Gemma 4 both extend this type but do not share fill or graphs.
 *
 * @since 1.1.0
 */
abstract sealed class CausalArchitecture implements ArchitectureProcessor
  permits Qwen3Processor, Gemma3Processor, Gemma4Processor, LlamaProcessor, Lfm2Processor {

  /**
   * Chat families are never embedding encoders.
   *
   * @return {@code false}
   * @since 1.1.0
   */
  @Override
  public final boolean isEmbedding() {
    return false;
  }

  /**
   * Hugging Face / ONNX {@code config.json} bind. Qwen3 and LFM2 override for GGUF.
   *
   * @since 1.1.0
   */
  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindHf(selected, ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  /**
   * Immutable causal decoder for this family.
   *
   * @param config  bound Hugging Face / GGUF-derived config
   * @param weights filled parameter bag
   * @return family graph ({@code *ForCausalLM})
   * @since 1.1.0
   */
  @Override
  public abstract CausalLM createCausal(final Config.HfConfig config, final WeightBag weights);
}
