package com.igormaznitsa.nanollvm.engine;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Per-{@link com.igormaznitsa.nanollvm.llm.LLM} KV page storage. Bound into {@link
 * com.igormaznitsa.nanollvm.internal.Context} for the duration of a forward pass.
 */
public final class KvCacheArena {

  private final Tensor[] kCaches;
  private final Tensor[] vCaches;

  public KvCacheArena(
      final int numLayers,
      final int numBlocks,
      final int blockSize,
      final int numKvHeads,
      final int headDim
  ) {
    if (numLayers <= 0) {
      throw new IllegalArgumentException("numLayers must be > 0");
    }
    if (numBlocks <= 0) {
      throw new IllegalArgumentException("numBlocks must be > 0");
    }
    this.kCaches = new Tensor[numLayers];
    this.vCaches = new Tensor[numLayers];
    for (int i = 0; i < numLayers; i++) {
      this.kCaches[i] = Tensor.zeros(numBlocks, blockSize, numKvHeads, headDim);
      this.vCaches[i] = Tensor.zeros(numBlocks, blockSize, numKvHeads, headDim);
    }
  }

  public int numLayers() {
    return this.kCaches.length;
  }

  public Tensor k(final int layerIndex) {
    return this.kCaches[this.requireLayer(layerIndex)];
  }

  public Tensor v(final int layerIndex) {
    return this.vCaches[this.requireLayer(layerIndex)];
  }

  private int requireLayer(final int layerIndex) {
    if (layerIndex < 0 || layerIndex >= this.kCaches.length) {
      throw new IllegalArgumentException(
          "layerIndex out of range: " + layerIndex + " (layers=" + this.kCaches.length + ")");
    }
    return layerIndex;
  }

  @Override
  public String toString() {
    requireNonNull(this.kCaches);
    return "KvCacheArena[layers=%d, blocks=%d]".formatted(
        this.kCaches.length,
        this.kCaches[0].size(0));
  }
}
