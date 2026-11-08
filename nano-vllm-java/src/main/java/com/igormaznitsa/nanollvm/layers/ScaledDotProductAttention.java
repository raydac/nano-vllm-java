package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Multi-head attention where query length and key/value length may differ (encoder self-attn,
 * decoder causal self-attn, and decoder-encoder cross-attn). A sequence of {@value #MATMUL_KEYS}
 * keys or more is one matmul per head, and the heads run together on the matmul pool. Shorter
 * sequences stay one dot per key.
 *
 * @since 1.3.0
 */
public final class ScaledDotProductAttention {

  private static final int MATMUL_KEYS = 32;
  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final int LANE = SPECIES.length();

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
    return this.forward(q, k, v, context, 0);
  }

  /**
   * Attention with an absolute start index for causal masking (incremental decode).
   *
   * <p>When {@code causal} and query row {@code i} has absolute position
   * {@code queryStartPosition + i}, it may attend only to keys {@code 0 .. position}. Prefill uses
   * {@code queryStartPosition = 0}. A single new decode token at absolute index {@code t} passes
   * {@code queryStartPosition = t} with {@code qLen = 1} and {@code kvLen = t + 1}.
   *
   * @param q                  queries {@code [qLen, numHeads * headDim]}
   * @param k                  keys {@code [kvLen, numHeads * headDim]}
   * @param v                  values {@code [kvLen, numHeads * headDim]}
   * @param context            step context, or {@code null} for sequential
   * @param queryStartPosition absolute index of query row 0 ({@code >= 0})
   * @return {@code [qLen, numHeads * headDim]}
   * @since 1.3.0
   */
  public Tensor forward(
    final Tensor q,
    final Tensor k,
    final Tensor v,
    final Context context,
    final int queryStartPosition
  ) {
    requireNonNull(q, "q");
    requireNonNull(k, "k");
    requireNonNull(v, "v");
    if (queryStartPosition < 0) {
      throw new IllegalArgumentException("queryStartPosition must be >= 0");
    }
    int qLen = q.size(0);
    int kvLen = k.size(0);
    if (v.size(0) != kvLen) {
      throw new IllegalArgumentException("k/v sequence lengths must match");
    }
    int width = this.numHeads * this.headDim;
    if (q.size(1) != width || k.size(1) != width || v.size(1) != width) {
      throw new IllegalArgumentException("q/k/v last dim must be numHeads*headDim");
    }
    if (this.causal && kvLen < queryStartPosition + qLen) {
      throw new IllegalArgumentException("causal attention requires kvLen >= query end");
    }
    Tensor out = Tensor.zeros(qLen, width);
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    if (kvLen >= MATMUL_KEYS) {
      this.attendByMatmul(q, k, v, out, qLen, kvLen, width, queryStartPosition, runtime);
      return out;
    }
    int work = this.numHeads * qLen;
    int cost = work * Math.max(kvLen, 1);
    if (cost < 256) {
      this.attendJobs(0, work, q, k, v, out, qLen, kvLen, width, queryStartPosition);
      return out;
    }
    runtime.parallelRanges(
      work,
      (start, end) -> this.attendJobs(
        start, end, q, k, v, out, qLen, kvLen, width, queryStartPosition));
    return out;
  }

  private void attendByMatmul(
    final Tensor q,
    final Tensor k,
    final Tensor v,
    final Tensor out,
    final int qLen,
    final int kvLen,
    final int width,
    final int queryStartPosition,
    final MatmulRuntime runtime
  ) {
    float[] qd = q.data();
    float[] kd = k.data();
    float[] vd = v.data();
    float[] od = out.data();
    int qOff = q.offset();
    int kOff = k.offset();
    int vOff = v.offset();
    int headDim = this.headDim;
    runtime.parallelRanges(this.numHeads, (start, end) -> {
      float[] query = new float[qLen * headDim];
      float[] key = new float[kvLen * headDim];
      float[] scores = new float[qLen * kvLen];
      float[] valueWeight = new float[headDim * kvLen];
      float[] mixed = new float[qLen * headDim];
      for (int head = start; head < end; head++) {
        int headBase = head * headDim;
        this.packHead(qd, qOff, query, qLen, width, headBase);
        this.packHead(kd, kOff, key, kvLen, width, headBase);
        runtime.linear(query, 0, key, 0, null, scores, 0, qLen, headDim, kvLen);
        this.softmaxScores(scores, qLen, kvLen, queryStartPosition);
        this.transposeValue(vd, vOff, valueWeight, kvLen, width, headBase);
        runtime.linear(scores, 0, valueWeight, 0, null, mixed, 0, qLen, kvLen, headDim);
        this.writeHead(mixed, od, qLen, width, headBase);
      }
    });
  }

  private void packHead(
    final float[] source,
    final int sourceOffset,
    final float[] packed,
    final int rows,
    final int width,
    final int headBase
  ) {
    int headDim = this.headDim;
    for (int row = 0; row < rows; row++) {
      System.arraycopy(
        source, sourceOffset + row * width + headBase, packed, row * headDim, headDim);
    }
  }

  private void softmaxScores(
    final float[] scores,
    final int qLen,
    final int kvLen,
    final int queryStartPosition
  ) {
    for (int row = 0; row < qLen; row++) {
      int lastKey = this.causal
        ? Math.min(queryStartPosition + row, kvLen - 1)
        : kvLen - 1;
      if (lastKey == kvLen - 1 && kvLen >= LANE) {
        this.softmaxDense(scores, row * kvLen, kvLen);
      } else {
        this.softmaxRow(scores, row, kvLen, lastKey);
      }
    }
  }

  private void softmaxDense(final float[] scores, final int base, final int kvLen) {
    FloatVector scale = FloatVector.broadcast(SPECIES, this.scale);
    FloatVector maxV = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
    int upper = SPECIES.loopBound(kvLen);
    int i = 0;
    for (; i < upper; i += LANE) {
      FloatVector scaled = FloatVector.fromArray(SPECIES, scores, base + i).mul(scale);
      scaled.intoArray(scores, base + i);
      maxV = maxV.max(scaled);
    }
    float max = maxV.reduceLanes(VectorOperators.MAX);
    for (; i < kvLen; i++) {
      float score = scores[base + i] * this.scale;
      scores[base + i] = score;
      if (score > max) {
        max = score;
      }
    }
    FloatVector maxB = FloatVector.broadcast(SPECIES, max);
    FloatVector sumV = FloatVector.zero(SPECIES);
    float sum = 0f;
    i = 0;
    for (; i < upper; i += LANE) {
      FloatVector weight = FloatVector.fromArray(SPECIES, scores, base + i)
        .sub(maxB)
        .lanewise(VectorOperators.EXP);
      weight.intoArray(scores, base + i);
      sumV = sumV.add(weight);
    }
    sum += sumV.reduceLanes(VectorOperators.ADD);
    for (; i < kvLen; i++) {
      float weight = (float) Math.exp(scores[base + i] - max);
      scores[base + i] = weight;
      sum += weight;
    }
    float inv = 1f / sum;
    FloatVector invB = FloatVector.broadcast(SPECIES, inv);
    i = 0;
    for (; i < upper; i += LANE) {
      FloatVector.fromArray(SPECIES, scores, base + i).mul(invB).intoArray(scores, base + i);
    }
    for (; i < kvLen; i++) {
      scores[base + i] *= inv;
    }
  }

  private void softmaxRow(
    final float[] scores, final int row, final int kvLen, final int lastKey
  ) {
    int base = row * kvLen;
    float max = Float.NEGATIVE_INFINITY;
    for (int key = 0; key <= lastKey; key++) {
      float score = scores[base + key] * this.scale;
      scores[base + key] = score;
      if (score > max) {
        max = score;
      }
    }
    float sum = 0f;
    for (int key = 0; key <= lastKey; key++) {
      float weight = (float) Math.exp(scores[base + key] - max);
      scores[base + key] = weight;
      sum += weight;
    }
    float inv = 1f / sum;
    for (int key = 0; key <= lastKey; key++) {
      scores[base + key] *= inv;
    }
    for (int key = lastKey + 1; key < kvLen; key++) {
      scores[base + key] = 0f;
    }
  }

  private void transposeValue(
    final float[] values,
    final int valueOffset,
    final float[] weight,
    final int kvLen,
    final int width,
    final int headBase
  ) {
    int headDim = this.headDim;
    for (int row = 0; row < kvLen; row++) {
      int src = valueOffset + row * width + headBase;
      for (int lane = 0; lane < headDim; lane++) {
        weight[lane * kvLen + row] = values[src + lane];
      }
    }
  }

  private void writeHead(
    final float[] mixed,
    final float[] dest,
    final int qLen,
    final int width,
    final int headBase
  ) {
    int headDim = this.headDim;
    for (int row = 0; row < qLen; row++) {
      System.arraycopy(mixed, row * headDim, dest, row * width + headBase, headDim);
    }
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
    final int width,
    final int queryStartPosition
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
      int lastKey = this.causal
        ? Math.min(queryStartPosition + i, kvLen - 1)
        : kvLen - 1;
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
