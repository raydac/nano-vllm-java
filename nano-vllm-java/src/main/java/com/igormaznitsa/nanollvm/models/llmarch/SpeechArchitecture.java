package com.igormaznitsa.nanollvm.models.llmarch;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.SpeechToText;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Template for speech families: Hugging Face bind by default, {@link #createSpeech} required,
 * never a chat or embedding graph.
 *
 * @since 1.3.0
 */
abstract sealed class SpeechArchitecture implements ArchitectureProcessor permits WhisperProcessor {

  @Override
  public final boolean isEmbedding() {
    return false;
  }

  @Override
  public final boolean isSpeech() {
    return true;
  }

  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindHf(selected, ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  @Override
  public abstract SpeechToText createSpeech(final Config.HfConfig config, final WeightBag weights);
}
