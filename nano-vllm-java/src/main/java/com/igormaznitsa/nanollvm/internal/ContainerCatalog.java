package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.ModelSupport;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture-agnostic snapshot of one weight container (GGUF file, HF folder, ONNX). Transport
 * fills this from on-disk layout; {@link ModelBinding} decides whether a supported graph can be
 * built from the metadata and tensor names.
 */
public record ContainerCatalog(
  ModelSupport.Source source,
  String label,
  String architectureHint,
  Map<String, Object> metadata,
  Set<String> tensorNames
) {

  public ContainerCatalog {
    requireNonNull(source, "source");
    requireNonNull(label, "label");
    architectureHint = architectureHint == null ? "" : architectureHint;
    metadata = Map.copyOf(requireNonNull(metadata, "metadata"));
    tensorNames = Set.copyOf(requireNonNull(tensorNames, "tensorNames"));
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
