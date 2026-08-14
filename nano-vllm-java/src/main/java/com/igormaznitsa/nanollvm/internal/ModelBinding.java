package com.igormaznitsa.nanollvm.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoderFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;

import java.util.List;

/**
 * Model layer: given a {@link ContainerCatalog} from transport, bind a supported architecture
 * (or fail) and produce {@link Config.HfConfig} plus the weight names that graph expects.
 */
public final class ModelBinding {

  private ModelBinding() {
  }

  public static BoundModel bindHf(final Config.HfConfig config, final ModelSupport.Source source) {
    ModelSupport.Selection selected =
      ModelSupport.require(requireNonNull(config, "config"), requireNonNull(source, "source"));
    WeightSchema schema = selected.isEmbedding()
      ? EmbeddingEncoderFactory.schema(config)
      : CausalLMFactory.schema(config);
    return new BoundModel(selected, config, schema);
  }

  public static BoundModel bindGguf(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    ModelSupport.Selection selected = ModelSupport.requireGguf(catalog.architectureHint());
    Config.HfConfig config = switch (selected.architectureId()) {
      case ARCH_QWEN3 -> GgufConfigs.qwen3(catalog);
      case ARCH_LFM2 -> GgufConfigs.lfm2(catalog);
      case ARCH_BERT -> GgufConfigs.bert(catalog);
      default -> throw new UnsupportedModelException(
        "GGUF architecture '%s' has no model binding."
          .formatted(catalog.architectureHint())
          + System.lineSeparator() + System.lineSeparator() + ModelSupport.CATALOG,
        catalog.architectureHint(),
        List.of());
    };
    WeightSchema schema = WeightSchema.forGguf(selected.architectureId(), config);
    BoundModel bound = new BoundModel(selected, config, schema);
    bound.requireCatalogTensors(catalog);
    return bound;
  }

  public record BoundModel(
    ModelSupport.Selection selection,
    Config.HfConfig config,
    WeightSchema schema
  ) {

    public BoundModel {
      requireNonNull(selection, "selection");
      requireNonNull(config, "config");
      requireNonNull(schema, "schema");
    }

    public void requireCatalogTensors(final ContainerCatalog catalog) {
      requireNonNull(catalog, "catalog");
      for (String required : this.schema.expectedParameters()) {
        if (!catalog.hasTensor(required)) {
          throw new IllegalStateException("missing required GGUF weight: " + required);
        }
      }
    }

    public void requireLoadedWeights(final WeightBag weights) {
      requireNonNull(weights, "weights");
      for (String required : this.schema.expectedParameters()) {
        if (!weights.has(required)) {
          throw new IllegalStateException("missing required GGUF weight: " + required);
        }
      }
    }
  }
}
