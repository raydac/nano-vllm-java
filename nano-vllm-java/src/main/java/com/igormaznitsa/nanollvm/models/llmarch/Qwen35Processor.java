package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3_5;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.Qwen35ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Qwen3.5 text chat from a Hugging Face safetensors folder or a GGUF file
 * ({@code general.architecture=qwen35}). Vision tensors are ignored.
 * Decisions go through {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#toJev()}.
 *
 * @since 1.5.0
 */
final class Qwen35Processor extends CausalArchitecture {

  static final Qwen35Processor INSTANCE = new Qwen35Processor();

  private Qwen35Processor() {
  }

  @Override
  public String architectureId() {
    return ARCH_QWEN3_5;
  }

  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindDualSource(catalog, selected, GgufConfigs::qwen35);
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new Qwen35ForCausalLM(config, weights);
  }
}
