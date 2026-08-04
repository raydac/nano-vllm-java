package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.Map;
import java.util.Optional;

/**
 * Immutable name → weight tensor map produced by {@link com.igormaznitsa.nanollvm.internal.ModelLoader}
 * before a {@link CausalLM} is constructed.
 */
public final class WeightBag {

  private final Map<String, Tensor> byName;

  public WeightBag(Map<String, Tensor> byName) {
    this.byName = Map.copyOf(requireNonNull(byName, "byName"));
  }

  public Tensor require(String name) {
    Tensor tensor = this.byName.get(name);
    if (tensor == null) {
      throw new IllegalArgumentException("missing weight: " + name);
    }
    return tensor;
  }

  public Optional<Tensor> find(String name) {
    return Optional.ofNullable(this.byName.get(name));
  }

  public boolean has(String name) {
    return this.byName.containsKey(name);
  }

  public int size() {
    return this.byName.size();
  }
}
