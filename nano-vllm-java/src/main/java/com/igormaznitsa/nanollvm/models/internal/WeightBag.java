package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable name → weight map. Entries are either dense float {@link Tensor}s (HF safetensors
 * or unpacked GGUF), {@link PackedWeight} (GGUF kept quantized until matmul / embedding), or
 * {@link GemmaQatWeight} (Gemma 4 packed int2/4/8).
 */
public final class WeightBag {

  private final Map<String, Object> byName;
  private final Map<String, ConvLayout> convLayouts;

  public WeightBag(final Map<String, ?> byName) {
    this(byName, Map.of());
  }

  public WeightBag(final Map<String, ?> byName, final Map<String, ConvLayout> convLayouts) {
    this.byName = Map.copyOf(requireNonNull(byName, "byName"));
    this.convLayouts = Map.copyOf(requireNonNull(convLayouts, "convLayouts"));
    for (Map.Entry<String, Object> entry : this.byName.entrySet()) {
      Object value = entry.getValue();
      if (!(value instanceof Tensor)
        && !(value instanceof PackedWeight)
        && !(value instanceof GemmaQatWeight)) {
        throw new IllegalArgumentException(
          "weight '" + entry.getKey() + "' must be Tensor, PackedWeight, or GemmaQatWeight");
      }
    }
  }

  public Optional<ConvLayout> convLayout(final String weightName) {
    return Optional.ofNullable(this.convLayouts.get(requireNonNull(weightName, "weightName")));
  }

  public int convDilation(final String weightName, final int fallback) {
    ConvLayout layout = this.convLayouts.get(requireNonNull(weightName, "weightName"));
    return layout == null ? fallback : layout.dilationOr(fallback);
  }

  private static boolean isPackedValue(final Object value) {
    return value instanceof PackedWeight || value instanceof GemmaQatWeight;
  }

  public Tensor require(final String name) {
    Object weight = this.requireRaw(name);
    if (weight instanceof Tensor tensor) {
      return tensor;
    }
    if (weight instanceof PackedWeight packed) {
      return packed.materialize();
    }
    return ((GemmaQatWeight) weight).materialize();
  }

  public PackedWeight requirePacked(final String name) {
    Object weight = this.requireRaw(name);
    if (weight instanceof PackedWeight packed) {
      return packed;
    }
    throw new IllegalArgumentException("weight is not packed GGUF: " + name);
  }

  public GemmaQatWeight requireQat(final String name) {
    Object weight = this.requireRaw(name);
    if (weight instanceof GemmaQatWeight qat) {
      return qat;
    }
    throw new IllegalArgumentException("weight is not Gemma QAT: " + name);
  }

  public Optional<PackedWeight> findPacked(final String name) {
    Object weight = this.byName.get(name);
    return weight instanceof PackedWeight packed ? Optional.of(packed) : Optional.empty();
  }

  public Optional<Tensor> find(final String name) {
    Object weight = this.byName.get(name);
    if (weight instanceof Tensor tensor) {
      return Optional.of(tensor);
    }
    if (weight instanceof PackedWeight packed) {
      return Optional.of(packed.materialize());
    }
    if (weight instanceof GemmaQatWeight qat) {
      return Optional.of(qat.materialize());
    }
    return Optional.empty();
  }

  public boolean has(final String name) {
    return this.byName.containsKey(name);
  }

  public Optional<GemmaQatWeight> findQat(final String name) {
    Object weight = this.byName.get(name);
    return weight instanceof GemmaQatWeight qat ? Optional.of(qat) : Optional.empty();
  }

  public boolean isPacked(final String name) {
    Object weight = this.byName.get(name);
    return weight instanceof PackedWeight || weight instanceof GemmaQatWeight;
  }

  public boolean hasPacked() {
    return this.byName.values().stream().anyMatch(WeightBag::isPackedValue);
  }

  public boolean hasGemmaQat() {
    return this.byName.values().stream().anyMatch(GemmaQatWeight.class::isInstance);
  }

  /**
   * Returns this bag if already dense; otherwise a new bag with every {@link PackedWeight}
   * expanded to float32 {@link Tensor}s (keeps this instance and its packed bytes unchanged).
   * Gemma 4 QAT entries cannot be expanded (PLE tables are multi-GB).
   */
  public WeightBag asDense() {
    if (this.hasGemmaQat()) {
      throw new IllegalStateException(
        "Gemma 4 QAT weights stay packed and cannot be unpacked to float32");
    }
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
    return new WeightBag(dense, this.convLayouts);
  }

  public int size() {
    return this.byName.size();
  }

  public Set<String> names() {
    return this.byName.keySet();
  }

  /**
   * Like {@link #asDense()}, but releases each packed byte payload immediately after materializing
   * that tensor so peak RAM stays near one full copy (not packed + float32 together).
   */
  public WeightBag asDenseReleasingPacked() {
    if (this.hasGemmaQat()) {
      throw new IllegalStateException(
        "Gemma 4 QAT weights stay packed and cannot be unpacked to float32");
    }
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
    return new WeightBag(dense, this.convLayouts);
  }

  private Object requireRaw(final String name) {
    Object weight = this.byName.get(name);
    if (weight == null) {
      throw new IllegalArgumentException("missing weight: " + name);
    }
    return weight;
  }

  /**
   * Drops packed GGUF / Gemma QAT payloads so quantized byte heaps become GC-eligible.
   * Dense {@link Tensor} entries stay reachable until callers drop this bag.
   */
  public void releaseResources() {
    this.byName.values().stream()
      .filter(PackedWeight.class::isInstance)
      .map(PackedWeight.class::cast)
      .forEach(PackedWeight::releasePackedBytes);
    this.byName.values().stream()
      .filter(GemmaQatWeight.class::isInstance)
      .map(GemmaQatWeight.class::cast)
      .forEach(GemmaQatWeight::releasePackedBytes);
  }
}
