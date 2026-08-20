package com.igormaznitsa.nanollvm.engine;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-LLM rolling short-convolution state for hybrid short-conv models (LFM2), keyed by
 * {@link Sequence#seqId()}.
 *
 * <p>Each sequence holds one {@code float[hiddenSize * (kernelSize - 1)]} row per layer — the
 * causal left-padding {@link com.igormaznitsa.nanollvm.layers.ShortConv} reads and writes.
 * {@link #state(int, int)} allocates on first touch. {@link Scheduler} calls {@link #clear(int)}
 * through {@link Transformer#clearConvState(int)} when a sequence's KV is released (finish,
 * preempt, cancel).
 *
 * <p>The backing map is concurrent so a release can drop a row without iterating the generate
 * thread's local structures; this is still not a concurrent generate API.
 *
 * @see Transformer
 * @see com.igormaznitsa.nanollvm.layers.ShortConv
 */
public final class ConvStateArena implements AutoCloseable {

  private final Map<Integer, float[][]> bySeqId = new ConcurrentHashMap<>();
  private final int numLayers;
  private final int hiddenSize;
  private final int stateLen;

  /**
   * @param numLayers  transformer layers ({@code > 0})
   * @param hiddenSize model width ({@code > 0})
   * @param kernelSize short-conv kernel ({@code >= 2}); stored state is {@code kernelSize - 1}
   * @throws IllegalArgumentException if any bound is violated
   */
  public ConvStateArena(final int numLayers, final int hiddenSize, final int kernelSize) {
    if (numLayers <= 0) {
      throw new IllegalArgumentException("numLayers must be > 0");
    }
    if (hiddenSize <= 0) {
      throw new IllegalArgumentException("hiddenSize must be > 0");
    }
    if (kernelSize < 2) {
      throw new IllegalArgumentException("kernelSize must be >= 2");
    }
    this.numLayers = numLayers;
    this.hiddenSize = hiddenSize;
    this.stateLen = kernelSize - 1;
  }

  /**
   * Live conv state for {@code seqId} at {@code layerIndex}. Created on first access; {@link
   * com.igormaznitsa.nanollvm.layers.ShortConv} mutates the returned array in place.
   *
   * @param seqId      {@link Sequence#seqId()}
   * @param layerIndex transformer layer
   * @return {@code hiddenSize * stateLen} floats
   * @throws IllegalArgumentException if {@code layerIndex} is out of range
   */
  public float[] state(final int seqId, final int layerIndex) {
    float[][] layers = this.bySeqId.computeIfAbsent(seqId, id -> this.newSeqStates());
    return layers[this.requireLayer(layerIndex)];
  }

  /**
   * Drops all layers for {@code seqId} (no-op if that id was never touched).
   *
   * @param seqId sequence to forget
   */
  public void clear(final int seqId) {
    this.bySeqId.remove(seqId);
  }

  /**
   * Drops every sequence. Used by {@link #close()}.
   */
  public void clearAll() {
    this.bySeqId.clear();
  }

  /**
   * Same as {@link #clearAll()}.
   */
  @Override
  public void close() {
    this.clearAll();
  }

  private float[][] newSeqStates() {
    float[][] layers = new float[this.numLayers][];
    int row = this.hiddenSize * this.stateLen;
    for (int i = 0; i < this.numLayers; i++) {
      layers[i] = new float[row];
    }
    return layers;
  }

  private int requireLayer(final int layerIndex) {
    if (layerIndex < 0 || layerIndex >= this.numLayers) {
      throw new IllegalArgumentException("layerIndex out of range: " + layerIndex);
    }
    return layerIndex;
  }

  /**
   * Layer count, live sequence count, and per-layer state length.
   *
   * @return short debug summary
   */
  @Override
  public String toString() {
    return "ConvStateArena[layers=%d, seqs=%d, stateLen=%d]".formatted(
      this.numLayers, this.bySeqId.size(), this.stateLen);
  }
}
