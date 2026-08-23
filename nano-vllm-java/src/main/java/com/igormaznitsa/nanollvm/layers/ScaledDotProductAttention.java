package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Arrays;

/**
 * Multi-head attention where query length and key/value length may differ (encoder self-attn,
 * decoder causal self-attn, and decoder-encoder cross-attn).
 *
 * @since 1.3.0
 */
public final class ScaledDotProductAttention {

  private final int numHeads;
  private final int headDim;
  private final float scale;
  private final boolean causal;

  /**
   * Packed multi-head attention.
   *
   * @param numHeads query/key/value heads (same count)
   * @param headDim  per-head width
   * @param scale    score multiplier, typically {@code 1/√headDim}
   * @param causal   when {@code true}, query {@code i} may attend only to keys {@code j <= i}
   */
  public ScaledDotProductAttention(
    final int numHeads,
    final int headDim,
    final float scale,
    final boolean causal
  ) {
    if (numHeads <= 0 || headDim <= 0) {
      throw new IllegalArgumentException("numHeads and headDim must be > 0");
    }
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.scale = scale;
    this.causal = causal;
  }

  /**
   * Sequential attention.
   *
   * @param q queries {@code [qLen, numHeads * headDim]}
   * @param k keys {@code [kvLen, numHeads * headDim]}
   * @param v values {@code [kvLen, numHeads * headDim]}
   * @return {@code [qLen, numHeads * headDim]}
   */
  public Tensor forward(final Tensor q, final Tensor k, final Tensor v) {
    return this.forward(q, k, v, null);
  }

  /**
   * Attention that may fan out head×query jobs on {@code context}'s matmul pool.
   *
   * @param q       queries {@code [qLen, numHeads * headDim]}
   * @param k       keys {@code [kvLen, numHeads * headDim]}
   * @param v       values {@code [kvLen, numHeads * headDim]}
   * @param context step context, or {@code null} for sequential
   * @return {@code [qLen, numHeads * headDim]}
   */
  public Tensor forward(final Tensor q, final Tensor k, final Tensor v, final Context context) {
    requireNonNull(q, "q");
    requireNonNull(k, "k");
    requireNonNull(v, "v");
    int qLen = q.size(0);
    int kvLen = k.size(0);
    if (v.size(0) != kvLen) {
      throw new IllegalArgumentException("k/v sequence lengths must match");
    }
    int width = this.numHeads * this.headDim;
    if (q.size(1) != width || k.size(1) != width || v.size(1) != width) {
      throw new IllegalArgumentException("q/k/v last dim must be numHeads*headDim");
    }
    if (this.causal && kvLen < qLen) {
      throw new IllegalArgumentException("causal attention requires kvLen >= qLen");
    }
    Tensor out = Tensor.zeros(qLen, width);
    int work = this.numHeads * qLen;
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    int cost = work * Math.max(kvLen, 1);
    if (cost < 256) {
      this.attendJobs(0, work, q, k, v, out, qLen, kvLen, width);
      return out;
    }
    runtime.parallelRanges(
      work, (start, end) -> this.attendJobs(start, end, q, k, v, out, qLen, kvLen, width));
    return out;
  }

  private void attendJobs(
    final int jobStart,
    final int jobEnd,
    final Tensor q,
    final Tensor k,
    final Tensor v,
    final Tensor out,
    final int qLen,
    final int kvLen,
    final int width
  ) {
    float[] qd = q.data();
    float[] kd = k.data();
    float[] vd = v.data();
    float[] od = out.data();
    int qOff = q.offset();
    int kOff = k.offset();
    int vOff = v.offset();
    float[] scores = new float[kvLen];

    for (int job = jobStart; job < jobEnd; job++) {
      int h = job / qLen;
      int i = job % qLen;
      int headBase = h * this.headDim;
      float max = Float.NEGATIVE_INFINITY;
      int qi = qOff + i * width + headBase;
      int lastKey = this.causal ? i : kvLen - 1;
      Arrays.fill(scores, Float.NEGATIVE_INFINITY);
      for (int j = 0; j <= lastKey; j++) {
        int kj = kOff + j * width + headBase;
        float score = VectorMath.dot(qd, qi, kd, kj, this.headDim) * this.scale;
        scores[j] = score;
        if (score > max) {
          max = score;
        }
      }
      float sum = 0f;
      for (int j = 0; j <= lastKey; j++) {
        float e = (float) Math.exp(scores[j] - max);
        scores[j] = e;
        sum += e;
      }
      float inv = 1f / sum;
      int oi = i * width + headBase;
      Arrays.fill(od, oi, oi + this.headDim, 0f);
      for (int j = 0; j <= lastKey; j++) {
        VectorMath.axpy(od, oi, scores[j] * inv, vd, vOff + j * width + headBase, this.headDim);
      }
    }
  }
}
