package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;

/**
 * Selects an {@link ArchitectureProcessor} from a {@link ContainerCatalog} and asks it to bind
 * config + weight schema for that family.
 *
 * @since 1.1.0
 */
public final class ModelBinding {

  private ModelBinding() {
  }

  /**
   * Binds GGUF, safetensors, or ONNX from the catalog source.
   *
   * @param catalog transport snapshot
   * @return bound family, config, schema, and processor
   * @since 1.1.0
   */
  public static BoundModel bind(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    return switch (catalog.source()) {
      case GGUF -> bindGguf(catalog);
      case HF_SAFETENSORS, ONNX -> bindHfCatalog(catalog);
      case FASTTEXT -> throw new IllegalArgumentException(
        "fastText models are not bound through ContainerCatalog");
    };
  }

  /**
   * Binds a parsed Hugging Face config for safetensors or ONNX.
   *
   * @param config parsed {@code config.json}
   * @param source {@link ModelSupport.Source#HF_SAFETENSORS} or {@link ModelSupport.Source#ONNX}
   * @return bound family, config, schema, and processor
   * @since 1.1.0
   */
  public static BoundModel bindHf(final Config.HfConfig config, final ModelSupport.Source source) {
    ModelSupport.Selection selected =
      ModelSupport.require(requireNonNull(config, "config"), requireNonNull(source, "source"));
    return ArchitectureProcessors.of(selected.architectureId())
      .bindHf(selected, config, source);
  }

  /**
   * Binds a GGUF catalog using {@code general.architecture}.
   *
   * @param catalog GGUF snapshot
   * @return bound family, config, schema, and processor
   * @since 1.1.0
   */
  public static BoundModel bindGguf(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    ModelSupport.Selection selected = ModelSupport.requireGguf(catalog.architectureHint());
    return ArchitectureProcessors.of(selected.architectureId()).bind(catalog, selected);
  }

  private static BoundModel bindHfCatalog(final ContainerCatalog catalog) {
    return bindHf(ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  /**
   * Config, weight schema, and processor chosen for one catalog.
   *
   * @param selection family accepted by {@link ModelSupport}
   * @param config    Hugging Face or GGUF-derived config
   * @param schema    expected parameter names for fill
   * @param processor architecture that will fill and create the graph
   * @since 1.1.0
   */
  public record BoundModel(
    ModelSupport.Selection selection,
    Config.HfConfig config,
    WeightSchema schema,
    ArchitectureProcessor processor
  ) {

    public BoundModel {
      requireNonNull(selection, "selection");
      requireNonNull(config, "config");
      requireNonNull(schema, "schema");
      requireNonNull(processor, "processor");
    }

    /**
     * Fails if the GGUF catalog is missing a schema parameter name.
     *
     * @param catalog GGUF snapshot
     * @since 1.1.0
     */
    public void requireCatalogTensors(final ContainerCatalog catalog) {
      requireNonNull(catalog, "catalog");
      for (String required : this.schema.expectedParameters()) {
        if (!catalog.hasTensor(required)) {
          throw new IllegalStateException("missing required GGUF weight: " + required);
        }
      }
    }

    /**
     * Fails if the filled bag is missing a schema parameter name.
     *
     * @param weights bag produced by {@link ArchitectureProcessor#fill}
     * @since 1.1.0
     */
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
