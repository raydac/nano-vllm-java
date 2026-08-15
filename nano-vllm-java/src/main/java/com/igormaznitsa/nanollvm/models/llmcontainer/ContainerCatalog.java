package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture-agnostic snapshot of one weight container (GGUF file, HF safetensors, ONNX).
 * Transport fills this from container layout; {@link com.igormaznitsa.nanollvm.models.llmarch.ArchitectureProcessor}
 * decides whether a supported graph can be built from the metadata and tensor names.
 *
 * @param source           GGUF, safetensors, or ONNX
 * @param label            path or virtual name for logs
 * @param architectureHint GGUF {@code general.architecture} or HF {@code model_type}; may be blank
 * @param metadata         string/number GGUF keys, or {@link #META_CONFIG_JSON} for HF/ONNX
 * @param tensorNames      payload tensor ids in this container
 * @since 1.1.0
 */
public record ContainerCatalog(
  ModelSupport.Source source,
  String label,
  String architectureHint,
  Map<String, Object> metadata,
  Set<String> tensorNames
) {

  /**
   * Metadata key for Hugging Face / ONNX {@code config.json} text.
   *
   * @since 1.1.0
   */
  public static final String META_CONFIG_JSON = "config.json";

  public ContainerCatalog {
    requireNonNull(source, "source");
    requireNonNull(label, "label");
    architectureHint = architectureHint == null ? "" : architectureHint;
    metadata = Map.copyOf(requireNonNull(metadata, "metadata"));
    tensorNames = Set.copyOf(requireNonNull(tensorNames, "tensorNames"));
  }

  /**
   * Hugging Face or ONNX catalog from {@code config.json} plus tensor names.
   *
   * @since 1.1.0
   */
  public static ContainerCatalog ofHf(
    final ModelSupport.Source source,
    final String label,
    final String configJson,
    final Set<String> tensorNames
  ) {
    requireNonNull(configJson, "configJson");
    return new ContainerCatalog(
      source,
      label,
      modelTypeOf(configJson),
      Map.of(META_CONFIG_JSON, configJson),
      tensorNames);
  }

  public static String modelTypeOf(final String configJson) {
    Map<String, Object> root = Json.parseObject(requireNonNull(configJson, "configJson"));
    String type = Json.asString(root.get("model_type"));
    return type == null ? "" : type;
  }

  public static Set<String> namesOf(final Iterable<String> names) {
    Set<String> out = new LinkedHashSet<>();
    names.forEach(out::add);
    return Set.copyOf(out);
  }

  public boolean hasTensor(final String name) {
    return this.tensorNames.contains(name);
  }

  public boolean hasMeta(final String key) {
    return this.metadata.containsKey(key);
  }

  public String metaString(final String key, final String defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof String s ? s : defaultValue;
  }

  public int metaInt(final String key, final int defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof Number n ? n.intValue() : defaultValue;
  }

  public float metaFloat(final String key, final float defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof Number n ? n.floatValue() : defaultValue;
  }

  public List<String> metaStringArray(final String key) {
    Object value = this.metadata.get(key);
    if (!(value instanceof List<?> list)) {
      return List.of();
    }
    return list.stream().map(Object::toString).toList();
  }

  public List<Number> metaNumberArray(final String key) {
    Object value = this.metadata.get(key);
    if (!(value instanceof List<?> list)) {
      return List.of();
    }
    List<Number> out = new ArrayList<>(list.size());
    for (Object item : list) {
      if (item instanceof Number n) {
        out.add(n);
      }
    }
    return List.copyOf(out);
  }
}
