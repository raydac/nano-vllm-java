package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.LlamaForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;

/**
 * Llama-style chat (SmolLM2, Tiny-LLM) from Hugging Face safetensors or ONNX.
 *
 * @since 1.1.0
 */
final class LlamaProcessor extends CausalArchitecture {

  static final LlamaProcessor INSTANCE = new LlamaProcessor();

  private LlamaProcessor() {
  }

  @Override
  public String architectureId() {
    return ARCH_LLAMA;
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new LlamaForCausalLM(config, weights);
  }
}
