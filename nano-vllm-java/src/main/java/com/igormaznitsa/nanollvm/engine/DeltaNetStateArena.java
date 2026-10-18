package com.igormaznitsa.nanollvm.engine;

import java.util.HashMap;
import java.util.Map;

/**
 * Per-sequence Gated DeltaNet state for Qwen3.5 linear-attention layers, keyed by
 * {@link Sequence#seqId()}.
 *
 * <p>Each linear layer holds a conv window {@code convDim * (kernelSize - 1)} and a recurrent
 * matrix {@code valueHeads * keyHeadDim * valueHeadDim}. Full-attention layers have no rows.
 * {@link Scheduler} drops a sequence through {@link Transformer#clearConvState(int)} when its
 * KV is released.
 *
 * <p>The backing map is generate-thread confined.
 *
 * @since 1.5.0
 */
public final class DeltaNetStateArena implements AutoCloseable {

  private final Map<Integer, SeqState> bySeqId = new HashMap<>();
  private final int numLayers;
  private final boolean[] linearLayer;
  private final int convFloats;
  private final int recurrentFloats;

  /**
   * @param linearLayer     {@code true} for layers that run Gated DeltaNet
   * @param convDim         depthwise conv channels ({@code > 0})
   * @param kernelSize      conv kernel ({@code >= 2}); stored window is {@code kernelSize - 1}
   * @param recurrentFloats recurrent matrix length per linear layer ({@code > 0})
   */
  public DeltaNetStateArena(
    final boolean[] linearLayer,
    final int convDim,
    final int kernelSize,
    final int recurrentFloats
  ) {
    if (linearLayer == null || linearLayer.length == 0) {
      throw new IllegalArgumentException("linearLayer must not be empty");
    }
    if (convDim <= 0) {
      throw new IllegalArgumentException("convDim must be > 0");
    }
    if (kernelSize < 2) {
      throw new IllegalArgumentException("kernelSize must be >= 2");
    }
    if (recurrentFloats <= 0) {
      throw new IllegalArgumentException("recurrentFloats must be > 0");
    }
    this.numLayers = linearLayer.length;
    this.linearLayer = linearLayer.clone();
    this.convFloats = convDim * (kernelSize - 1);
    this.recurrentFloats = recurrentFloats;
  }

  /**
   * Conv window for {@code seqId} at a linear-attention layer. Created on first access.
   *
   * @return {@code convDim * (kernelSize - 1)} floats, oldest sample at tap 0
   */
  public float[] convState(final int seqId, final int layerIndex) {
    return this.state(seqId).conv[this.requireLinear(layerIndex)];
  }

  /**
   * Recurrent matrix for {@code seqId} at a linear-attention layer. Created on first access.
   *
   * @return {@code valueHeads * keyHeadDim * valueHeadDim} floats
   */
  public float[] recurrentState(final int seqId, final int layerIndex) {
    return this.state(seqId).recurrent[this.requireLinear(layerIndex)];
  }

  /**
   * Drops both conv and recurrent rows for {@code seqId}.
   */
  public void clear(final int seqId) {
    this.bySeqId.remove(seqId);
  }

  /**
   * Drops every sequence.
   */
  public void clearAll() {
    this.bySeqId.clear();
  }

  @Override
  public void close() {
    this.clearAll();
  }

  private SeqState state(final int seqId) {
    return this.bySeqId.computeIfAbsent(seqId, id -> this.newSeqState());
  }

  private SeqState newSeqState() {
    float[][] conv = new float[this.numLayers][];
    float[][] recurrent = new float[this.numLayers][];
    for (int i = 0; i < this.numLayers; i++) {
      if (!this.linearLayer[i]) {
        continue;
      }
      conv[i] = new float[this.convFloats];
      recurrent[i] = new float[this.recurrentFloats];
    }
    return new SeqState(conv, recurrent);
  }

  private int requireLinear(final int layerIndex) {
    if (layerIndex < 0 || layerIndex >= this.numLayers || !this.linearLayer[layerIndex]) {
      throw new IllegalArgumentException("not a linear-attention layer: " + layerIndex);
    }
    return layerIndex;
  }

  private record SeqState(float[][] conv, float[][] recurrent) {
  }
}
