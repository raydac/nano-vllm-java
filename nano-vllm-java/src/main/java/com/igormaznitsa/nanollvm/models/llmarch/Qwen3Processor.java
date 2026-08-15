package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.Qwen3ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Qwen3 chat: Hugging Face safetensors / ONNX, or GGUF ({@code general.architecture=qwen3}).
 *
 * @since 1.1.0
 */
final class Qwen3Processor extends CausalArchitecture {

  static final Qwen3Processor INSTANCE = new Qwen3Processor();

  private Qwen3Processor() {
  }

  @Override
  public String architectureId() {
    return ARCH_QWEN3;
  }

  /**
   * GGUF catalog uses {@link GgufConfigs#qwen3}; otherwise Hugging Face {@code config.json}.
   *
   * @since 1.1.0
   */
  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindDualSource(catalog, selected, GgufConfigs::qwen3);
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new Qwen3ForCausalLM(config, weights);
  }
}
