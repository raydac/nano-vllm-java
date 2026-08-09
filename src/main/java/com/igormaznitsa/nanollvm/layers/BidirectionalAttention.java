package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Full (non-causal) multi-head self-attention for encoder models. No KV cache.
 *
 * @since 1.1.0
 */
public final class BidirectionalAttention {

  private final int numHeads;
  private final int headDim;
  private final float scale;

  public BidirectionalAttention(final int numHeads, final int headDim, final float scale) {
    if (numHeads <= 0 || headDim <= 0) {
      throw new IllegalArgumentException("numHeads and headDim must be > 0");
    }
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.scale = scale;
  }

  public Tensor forward(final Tensor q, final Tensor k, final Tensor v) {
    requireNonNull(q, "q");
    requireNonNull(k, "k");
    requireNonNull(v, "v");
    int seq = q.size(0);
    if (k.size(0) != seq || v.size(0) != seq) {
      throw new IllegalArgumentException("q/k/v sequence lengths must match");
    }
    int qWidth = this.numHeads * this.headDim;
    if (q.size(1) != qWidth || k.size(1) != qWidth || v.size(1) != qWidth) {
      throw new IllegalArgumentException("q/k/v last dim must be numHeads*headDim");
    }

    Tensor out = Tensor.zeros(seq, qWidth);
    float[] qd = q.data();
    float[] kd = k.data();
    float[] vd = v.data();
    float[] od = out.data();
    int qOff = q.offset();
    int kOff = k.offset();
    int vOff = v.offset();
    float[] scores = new float[seq];

    for (int h = 0; h < this.numHeads; h++) {
      int headBase = h * this.headDim;
      for (int i = 0; i < seq; i++) {
        float max = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < seq; j++) {
          float dot = 0f;
          int qi = qOff + i * qWidth + headBase;
          int kj = kOff + j * qWidth + headBase;
          for (int d = 0; d < this.headDim; d++) {
            dot += qd[qi + d] * kd[kj + d];
          }
          float score = dot * this.scale;
          scores[j] = score;
          if (score > max) {
            max = score;
          }
        }
        float sum = 0f;
        for (int j = 0; j < seq; j++) {
          float e = (float) Math.exp(scores[j] - max);
          scores[j] = e;
          sum += e;
        }
        float inv = 1f / sum;
        int oi = i * qWidth + headBase;
        for (int d = 0; d < this.headDim; d++) {
          od[oi + d] = 0f;
        }
        for (int j = 0; j < seq; j++) {
          float w = scores[j] * inv;
          int vj = vOff + j * qWidth + headBase;
          for (int d = 0; d < this.headDim; d++) {
            od[oi + d] += w * vd[vj + d];
          }
        }
      }
    }
    return out;
  }
}
