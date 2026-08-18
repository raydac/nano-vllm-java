package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;

/**
 * Per-{@link com.igormaznitsa.nanollvm.llm.LLM} KV page storage. Bound into {@link
 * com.igormaznitsa.nanollvm.internal.Context} for the duration of a forward pass.
 */
public final class KvCacheArena implements AutoCloseable {

  private final Tensor[] kCaches;
  private final Tensor[] vCaches;

  public KvCacheArena(
    final int numLayers,
    final int numBlocks,
    final int blockSize,
    final int numKvHeads,
    final int headDim
  ) {
    this(numBlocks, blockSize, numKvHeads, fill(numLayers, headDim), fillTrue(numLayers));
  }

  public KvCacheArena(
    final int numBlocks,
    final int blockSize,
    final int numKvHeads,
    final int[] headDims,
    final boolean[] allocateLayer
  ) {
    if (headDims.length == 0) {
      throw new IllegalArgumentException("numLayers must be > 0");
    }
    if (numBlocks <= 0) {
      throw new IllegalArgumentException("numBlocks must be > 0");
    }
    if (headDims.length != allocateLayer.length) {
      throw new IllegalArgumentException("headDims and allocateLayer length mismatch");
    }
    this.kCaches = new Tensor[headDims.length];
    this.vCaches = new Tensor[headDims.length];
    for (int i = 0; i < headDims.length; i++) {
      if (allocateLayer[i]) {
        this.kCaches[i] = Tensor.zeros(numBlocks, blockSize, numKvHeads, headDims[i]);
        this.vCaches[i] = Tensor.zeros(numBlocks, blockSize, numKvHeads, headDims[i]);
      } else {
        this.kCaches[i] = Tensor.zeros(1);
        this.vCaches[i] = Tensor.zeros(1);
      }
    }
  }

  private static int[] fill(final int n, final int value) {
    int[] out = new int[n];
    Arrays.fill(out, value);
    return out;
  }

  private static boolean[] fillTrue(final int n) {
    boolean[] out = new boolean[n];
    Arrays.fill(out, true);
    return out;
  }

  public Tensor k(final int layerIndex) {
    return this.kCaches[this.requireLayer(layerIndex)];
  }

  public Tensor v(final int layerIndex) {
    return this.vCaches[this.requireLayer(layerIndex)];
  }

  @Override
  public void close() {
    Arrays.fill(this.kCaches, null);
    Arrays.fill(this.vCaches, null);
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
    Tensor first = this.kCaches.length == 0 ? null : this.kCaches[0];
    return "KvCacheArena[layers=%d, blocks=%d]".formatted(
      this.kCaches.length,
      first != null && first.ndim() >= 2 ? first.size(0) : 0);
  }
}
