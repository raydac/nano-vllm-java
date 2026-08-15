package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.Gemma4ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsTransport;
import java.io.IOException;

/**
 * Gemma 4 text (QAT mobile): Hugging Face safetensors only; packed int2/4/8, not GGUF or ONNX.
 *
 * @since 1.1.0
 */
final class Gemma4Processor extends CausalArchitecture {

  static final Gemma4Processor INSTANCE = new Gemma4Processor();

  private Gemma4Processor() {
  }

  @Override
  public String architectureId() {
    return ARCH_GEMMA4;
  }

  /**
   * Packed QAT decode via {@link Gemma4QatLoader}; rejects GGUF and ONNX.
   *
   * @throws IllegalStateException if {@code transport} is not safetensors
   * @since 1.1.0
   */
  @Override
  public WeightBag fill(
    final ContainerTransport transport,
    final BoundModel bound,
    final LlmListener io,
    final boolean allowUnpackGguf
  ) throws IOException {
    requireNonNull(transport, "transport");
    requireNonNull(bound, "bound");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    if (transport instanceof SafetensorsTransport safetensors) {
      return Gemma4QatLoader.fill(safetensors, bound, streams);
    }
    throw new IllegalStateException("Gemma 4 text loads from safetensors only");
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new Gemma4ForCausalLM(config, weights);
  }
}
