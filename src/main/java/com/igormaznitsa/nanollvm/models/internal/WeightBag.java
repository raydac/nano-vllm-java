package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Immutable name → weight map. Entries are either dense float {@link Tensor}s (HF safetensors
 * or unpacked GGUF) or {@link PackedWeight} (GGUF kept quantized until matmul / embedding).
 */
public final class WeightBag {

  private final Map<String, Object> byName;

  public WeightBag(final Map<String, ?> byName) {
    this.byName = Map.copyOf(requireNonNull(byName, "byName"));
    for (Map.Entry<String, Object> entry : this.byName.entrySet()) {
      Object value = entry.getValue();
      if (!(value instanceof Tensor) && !(value instanceof PackedWeight)) {
        throw new IllegalArgumentException(
          "weight '" + entry.getKey() + "' must be Tensor or PackedWeight");
      }
    }
  }

  public Tensor require(final String name) {
    Object weight = this.requireRaw(name);
    if (weight instanceof Tensor tensor) {
      return tensor;
    }
    return ((PackedWeight) weight).materialize();
  }

  public PackedWeight requirePacked(final String name) {
    Object weight = this.requireRaw(name);
    if (weight instanceof PackedWeight packed) {
      return packed;
    }
    throw new IllegalArgumentException("weight is dense float, not packed: " + name);
  }

  public Optional<Tensor> find(final String name) {
    Object weight = this.byName.get(name);
    if (weight instanceof Tensor tensor) {
      return Optional.of(tensor);
    }
    if (weight instanceof PackedWeight packed) {
      return Optional.of(packed.materialize());
    }
    return Optional.empty();
  }

  public Optional<PackedWeight> findPacked(final String name) {
    Object weight = this.byName.get(name);
    return weight instanceof PackedWeight packed ? Optional.of(packed) : Optional.empty();
  }

  public boolean has(final String name) {
    return this.byName.containsKey(name);
  }

  public boolean isPacked(final String name) {
    return this.byName.get(name) instanceof PackedWeight;
  }

  public boolean hasPacked() {
    return this.byName.values().stream().anyMatch(PackedWeight.class::isInstance);
  }

  /**
   * Returns this bag if already dense; otherwise a new bag with every {@link PackedWeight}
   * expanded to float32 {@link Tensor}s (keeps this instance and its packed bytes unchanged).
   */
  public WeightBag asDense() {
    if (!this.hasPacked()) {
      return this;
    }
    Map<String, Object> dense = new LinkedHashMap<>(this.byName.size());
    for (Map.Entry<String, Object> entry : this.byName.entrySet()) {
      Object value = entry.getValue();
      dense.put(
        entry.getKey(),
        value instanceof PackedWeight packed ? packed.materialize() : value);
    }
    return new WeightBag(dense);
  }

  /**
   * Like {@link #asDense()}, but releases each packed byte payload immediately after materializing
   * that tensor so peak RAM stays near one full copy (not packed + float32 together).
   */
  public WeightBag asDenseReleasingPacked() {
    if (!this.hasPacked()) {
      return this;
    }
    Map<String, Object> dense = new LinkedHashMap<>(this.byName.size());
    for (Map.Entry<String, Object> entry : this.byName.entrySet()) {
      Object value = entry.getValue();
      if (value instanceof PackedWeight packed) {
        Tensor tensor = packed.materialize();
        packed.releasePackedBytes();
        dense.put(entry.getKey(), tensor);
      } else {
        dense.put(entry.getKey(), value);
      }
    }
    return new WeightBag(dense);
  }

  public int size() {
    return this.byName.size();
  }

  private Object requireRaw(final String name) {
    Object weight = this.byName.get(name);
    if (weight == null) {
      throw new IllegalArgumentException("missing weight: " + name);
    }
    return weight;
  }
}
