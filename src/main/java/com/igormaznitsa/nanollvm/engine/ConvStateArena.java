package com.igormaznitsa.nanollvm.engine;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-LLM rolling short-convolution state for hybrid models (LFM2), keyed by sequence id.
 */
public final class ConvStateArena {

  private final Map<Integer, float[][]> bySeqId = new ConcurrentHashMap<>();
  private final int numLayers;
  private final int hiddenSize;
  private final int stateLen;

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

  public float[] state(final int seqId, final int layerIndex) {
    float[][] layers = this.bySeqId.computeIfAbsent(seqId, id -> this.newSeqStates());
    return layers[this.requireLayer(layerIndex)];
  }

  public void clear(final int seqId) {
    this.bySeqId.remove(seqId);
  }

  public void clearAll() {
    this.bySeqId.clear();
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

  @Override
  public String toString() {
    return "ConvStateArena[layers=%d, seqs=%d, stateLen=%d]".formatted(
      this.numLayers, this.bySeqId.size(), this.stateLen);
  }
}
