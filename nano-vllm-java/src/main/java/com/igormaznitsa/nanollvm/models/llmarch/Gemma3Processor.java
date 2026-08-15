package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.Gemma3ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;

/**
 * Gemma 3 / {@code gemma3_text} chat from Hugging Face safetensors or ONNX.
 *
 * @since 1.1.0
 */
final class Gemma3Processor extends CausalArchitecture {

  static final Gemma3Processor INSTANCE = new Gemma3Processor();

  private Gemma3Processor() {
  }

  @Override
  public String architectureId() {
    return ARCH_GEMMA3;
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new Gemma3ForCausalLM(config, weights);
  }
}
