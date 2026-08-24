package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_PIPER;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_WHISPER;
import static java.util.Objects.requireNonNull;


/**
 * Registry of {@link ArchitectureProcessor}s keyed by canonical architecture id.
 *
 * @since 1.1.0
 */
public final class ArchitectureProcessors {

  private ArchitectureProcessors() {
  }

  /**
   * Processor for {@code architectureId} ({@code qwen3}, {@code gemma3}, {@code gemma4},
   * {@code llama}, {@code lfm2}, {@code bert}, {@code whisper}, {@code piper}).
   *
   * @param architectureId canonical backend id
   * @return singleton processor for that family
   * @throws IllegalArgumentException if no processor is registered
   * @since 1.1.0
   */
  public static ArchitectureProcessor of(final String architectureId) {
    return switch (requireNonNull(architectureId, "architectureId")) {
      case ARCH_QWEN3 -> Qwen3Processor.INSTANCE;
      case ARCH_GEMMA3 -> Gemma3Processor.INSTANCE;
      case ARCH_GEMMA4 -> Gemma4Processor.INSTANCE;
      case ARCH_LLAMA -> LlamaProcessor.INSTANCE;
      case ARCH_LFM2 -> Lfm2Processor.INSTANCE;
      case ARCH_BERT -> BertProcessor.INSTANCE;
      case ARCH_WHISPER -> WhisperProcessor.INSTANCE;
      case ARCH_PIPER -> PiperProcessor.INSTANCE;
      default -> throw new IllegalArgumentException(
        "no architecture processor for '" + architectureId + "'");
    };
  }
}
