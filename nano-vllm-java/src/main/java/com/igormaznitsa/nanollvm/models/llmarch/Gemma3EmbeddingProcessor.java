package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3_EMBEDDING;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.Gemma3ForEmbedding;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Bidirectional Gemma 3 text embeddings from a Hugging Face safetensors folder.
 *
 * @since 1.5.0
 */
final class Gemma3EmbeddingProcessor extends EmbeddingArchitecture {

  static final Gemma3EmbeddingProcessor INSTANCE = new Gemma3EmbeddingProcessor();

  private Gemma3EmbeddingProcessor() {
  }

  @Override
  public String architectureId() {
    return ARCH_GEMMA3_EMBEDDING;
  }

  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindHf(selected, ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  @Override
  public EmbeddingEncoder createEmbedding(final Config.HfConfig config, final WeightBag weights) {
    return new Gemma3ForEmbedding(config, weights);
  }
}
