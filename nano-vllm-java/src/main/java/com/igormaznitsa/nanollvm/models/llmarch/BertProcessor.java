package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.BertForEmbedding;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * BERT sentence embeddings from GGUF or ONNX (not Hugging Face safetensors).
 *
 * @since 1.1.0
 */
final class BertProcessor extends EmbeddingArchitecture {

  static final BertProcessor INSTANCE = new BertProcessor();

  private BertProcessor() {
  }

  @Override
  public String architectureId() {
    return ARCH_BERT;
  }

  /**
   * GGUF catalog uses {@link GgufConfigs#bert}; otherwise ONNX {@code config.json}.
   *
   * @since 1.1.0
   */
  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindDualSource(catalog, selected, GgufConfigs::bert);
  }

  @Override
  public EmbeddingEncoder createEmbedding(final Config.HfConfig config, final WeightBag weights) {
    return new BertForEmbedding(config, weights);
  }
}
