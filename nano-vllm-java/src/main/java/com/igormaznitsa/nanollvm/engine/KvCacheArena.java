package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;

/**
 * Per-{@link com.igormaznitsa.nanollvm.llm.LLM} KV page storage. Bound into {@link
 * com.igormaznitsa.nanollvm.internal.Context} for the duration of a forward pass.
 *
 * <p>Each allocated layer holds K and V as {@code [numBlocks, blockSize, numKvHeads, headDim]}.
 * Physical slot {@code s} is {@code blockId * blockSize + offset} along the flattened
 * {@code [numBlocks * blockSize, numKvHeads, headDim]} layout attention reads. Gemma 4 shared-KV
 * consumer layers pass {@code allocateLayer[i] == false} and get a dummy tensor of {@code numel 1}
 * so {@link com.igormaznitsa.nanollvm.layers.Attention} skips the store and reads the producer
 * layer instead.
 *
 * <p>{@link #close()} drops tensor references so the pages can be GC'd. Does not zero the float
 * buffers first.
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; one arena per {@code LLM}, generate
 * thread only.
 *
 * @see Transformer
 * @see BlockManager
 */
public final class KvCacheArena implements AutoCloseable {

  private final Tensor[] kCaches;
  private final Tensor[] vCaches;

  /**
   * Uniform head width; every layer is allocated.
   *
   * @param numLayers  transformer layers
   * @param numBlocks  pages in the pool
   * @param blockSize  tokens per page
   * @param numKvHeads K/V heads
   * @param headDim    per-head width (same for all layers)
   * @throws IllegalArgumentException if {@code numLayers} or {@code numBlocks} is not {@code > 0}
   */
  public KvCacheArena(
    final int numLayers,
    final int numBlocks,
    final int blockSize,
    final int numKvHeads,
    final int headDim
  ) {
    this(numBlocks, blockSize, numKvHeads, fill(numLayers, headDim), fillTrue(numLayers));
  }

  /**
   * Per-layer head width and optional dummy pages (Gemma 4 KV sharing).
   *
   * @param numBlocks     pages in the pool
   * @param blockSize     tokens per page
   * @param numKvHeads    K/V heads
   * @param headDims      per-layer head width; length is the layer count
   * @param allocateLayer {@code false} → dummy {@code numel 1} tensor for that layer
   * @throws IllegalArgumentException if lengths disagree, {@code headDims} is empty, or
   *                                  {@code numBlocks <= 0}
   * @since 1.1.0
   */
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

  /**
   * Key cache for {@code layerIndex}. Dummy {@code numel 1} when the layer does not store KV.
   *
   * @param layerIndex transformer layer
   * @return K pages (or dummy)
   * @throws IllegalArgumentException if {@code layerIndex} is out of range
   */
  public Tensor k(final int layerIndex) {
    return this.kCaches[this.requireLayer(layerIndex)];
  }

  /**
   * Value cache for {@code layerIndex}. Dummy {@code numel 1} when the layer does not store KV.
   *
   * @param layerIndex transformer layer
   * @return V pages (or dummy)
   * @throws IllegalArgumentException if {@code layerIndex} is out of range
   */
  public Tensor v(final int layerIndex) {
    return this.vCaches[this.requireLayer(layerIndex)];
  }

  /**
   * Drops K/V tensor references so pages can be reclaimed. Further {@link #k(int)} / {@link #v(int)}
   * calls still index the array (now {@code null} entries).
   */
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

  /**
   * Layer count and page count of the first allocated K tensor (or {@code 0} pages if empty).
   *
   * @return short debug summary
   */
  @Override
  public String toString() {
    Tensor first = this.kCaches.length == 0 ? null : this.kCaches[0];
    return "KvCacheArena[layers=%d, blocks=%d]".formatted(
      this.kCaches.length,
      first != null && first.ndim() >= 2 ? first.size(0) : 0);
  }
}
