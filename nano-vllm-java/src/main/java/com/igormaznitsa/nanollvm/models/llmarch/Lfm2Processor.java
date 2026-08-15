package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.Lfm2ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufTransport;
import java.io.IOException;

/**
 * LFM2 hybrid chat from GGUF only ({@code general.architecture=lfm2}).
 *
 * @since 1.1.0
 */
final class Lfm2Processor extends CausalArchitecture {

  static final Lfm2Processor INSTANCE = new Lfm2Processor();

  private Lfm2Processor() {
  }

  @Override
  public String architectureId() {
    return ARCH_LFM2;
  }

  /**
   * GGUF metadata via {@link GgufConfigs#lfm2}.
   *
   * @since 1.1.0
   */
  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindGguf(selected, catalog, GgufConfigs.lfm2(catalog));
  }

  /**
   * GGUF payloads only.
   *
   * @throws IllegalStateException if {@code transport} is not GGUF
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
    if (transport instanceof GgufTransport gguf) {
      LlmListener streams = io == null ? LlmListeners.silent() : io;
      return ArchitectureFills.gguf(gguf, bound, streams, allowUnpackGguf);
    }
    throw new IllegalStateException("LFM2 loads from GGUF only");
  }

  @Override
  public CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    return new Lfm2ForCausalLM(config, weights);
  }
}
