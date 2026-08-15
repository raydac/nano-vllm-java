package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import java.io.IOException;

/**
 * Hands container payloads to the bound {@link ArchitectureProcessor} to fill a {@link WeightBag}.
 *
 * @since 1.1.0
 */
public final class ModelFill {

  private ModelFill() {
  }

  /**
   * Delegates to {@link ArchitectureProcessor#fill}.
   *
   * @param transport       open container matching the catalog used at bind
   * @param bound           result of {@link ModelBinding#bind}
   * @param io              load progress; {@code null} is treated as silent
   * @param allowUnpackGguf when {@code true}, GGUF packed tensors are expanded to float32
   * @return filled bag for graph construction
   * @throws IOException if a payload cannot be read
   * @since 1.1.0
   */
  public static WeightBag fill(
    final ContainerTransport transport,
    final BoundModel bound,
    final LlmListener io,
    final boolean allowUnpackGguf
  ) throws IOException {
    requireNonNull(transport, "transport");
    requireNonNull(bound, "bound");
    return bound.processor().fill(transport, bound, io, allowUnpackGguf);
  }
}
