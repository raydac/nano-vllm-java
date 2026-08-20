package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Arrays;

/**
 * Full (non-causal) multi-head self-attention for encoder models. No KV cache and no GQA:
 * {@code q}, {@code k}, and {@code v} share the same head count. BERT embeddings use this;
 * chat graphs use {@link Attention}.
 *
 * <p>Inputs are packed {@code [seq, numHeads * headDim]}. Every query position attends to every
 * key (no causal mask, no sliding window). Softmax is per head, per query row. Head×query jobs
 * are independent and run on {@link Context#matmul()} when the work is large enough.
 *
 * @since 1.1.0
 */
public final class BidirectionalAttention {

  private final int numHeads;
  private final int headDim;
  private final float scale;

  /**
   * Full self-attention with {@code numHeads} heads of width {@code headDim}.
   *
   * @param numHeads query/key/value heads (same count)
   * @param headDim  per-head width
   * @param scale    score multiplier, typically {@code 1/√headDim}
   * @throws IllegalArgumentException if {@code numHeads} or {@code headDim} is {@code <= 0}
   */
  public BidirectionalAttention(final int numHeads, final int headDim, final float scale) {
    if (numHeads <= 0 || headDim <= 0) {
      throw new IllegalArgumentException("numHeads and headDim must be > 0");
    }
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.scale = scale;
  }

  /**
   * Full self-attention over one sequence (sequential runtime).
   *
   * @param q queries {@code [seq, numHeads * headDim]}
   * @param k keys, same shape as {@code q}
   * @param v values, same shape as {@code q}
   * @return attended values, same shape as {@code q}
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if sequence lengths or last dims disagree
   */
  public Tensor forward(final Tensor q, final Tensor k, final Tensor v) {
    return this.forward(q, k, v, null);
  }

  /**
   * Full self-attention over one sequence. When {@code context} binds a parallel
   * {@link MatmulRuntime}, independent head×query jobs may run concurrently.
   *
   * @param q       queries {@code [seq, numHeads * headDim]}
   * @param k       keys, same shape as {@code q}
   * @param v       values, same shape as {@code q}
   * @param context step context (matmul pool), or {@code null} for sequential
   * @return attended values, same shape as {@code q}
   * @throws NullPointerException     if {@code q}, {@code k}, or {@code v} is {@code null}
   * @throws IllegalArgumentException if sequence lengths or last dims disagree
   */
  public Tensor forward(final Tensor q, final Tensor k, final Tensor v, final Context context) {
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
    int work = this.numHeads * seq;
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    int cost = work * Math.max(seq, 1);
    if (cost < 256) {
      this.attendJobs(0, work, q, k, v, out, seq, qWidth);
      return out;
    }
    runtime.parallelRanges(
      work, (start, end) -> this.attendJobs(start, end, q, k, v, out, seq, qWidth));
    return out;
  }

  private void attendJobs(
    final int jobStart,
    final int jobEnd,
    final Tensor q,
    final Tensor k,
    final Tensor v,
    final Tensor out,
    final int seq,
    final int qWidth
  ) {
    float[] qd = q.data();
    float[] kd = k.data();
    float[] vd = v.data();
    float[] od = out.data();
    int qOff = q.offset();
    int kOff = k.offset();
    int vOff = v.offset();
    float[] scores = new float[seq];

    for (int job = jobStart; job < jobEnd; job++) {
      int h = job / seq;
      int i = job % seq;
      int headBase = h * this.headDim;
      float max = Float.NEGATIVE_INFINITY;
      int qi = qOff + i * qWidth + headBase;
      for (int j = 0; j < seq; j++) {
        int kj = kOff + j * qWidth + headBase;
        float score = VectorMath.dot(qd, qi, kd, kj, this.headDim) * this.scale;
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
      Arrays.fill(od, oi, oi + this.headDim, 0f);
      for (int j = 0; j < seq; j++) {
        VectorMath.axpy(od, oi, scores[j] * inv, vd, vOff + j * qWidth + headBase, this.headDim);
      }
    }
  }
}
