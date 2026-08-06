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

  public WeightBag(final Map<String, Tensor> byName) {
    this.byName = Map.copyOf(requireNonNull(byName, "byName"));
  }

  public Tensor require(final String name) {
    Tensor tensor = this.byName.get(name);
    if (tensor == null) {
      throw new IllegalArgumentException("missing weight: " + name);
    }
    return tensor;
  }

  public Optional<Tensor> find(final String name) {
    return Optional.ofNullable(this.byName.get(name));
  }

  public boolean has(final String name) {
    return this.byName.containsKey(name);
  }

  public int size() {
    return this.byName.size();
  }
}
