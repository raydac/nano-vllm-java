package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Arrays;

/**
 * Bidirectional grouped-query attention for an encoder. No KV cache. A positive window is an
 * exclusive distance: query {@code i} attends to key {@code j} when {@code |i - j|} is smaller
 * than that distance. Zero attends to every key.
 *
 * @since 1.5.0
 */
public final class GroupedEncoderAttention {

  private final int numHeads;
  private final int numKvHeads;
  private final int headDim;
  private final int repeats;
  private final float scale;
  private final int windowExclusive;

  /**
   * @param numHeads        query heads
   * @param numKvHeads      key and value heads; {@code numHeads} must be a multiple of this
   * @param headDim         per-head width
   * @param scale           score multiplier
   * @param windowExclusive exclusive key distance, or {@code 0} for every key
   */
  public GroupedEncoderAttention(
    final int numHeads,
    final int numKvHeads,
    final int headDim,
    final float scale,
    final int windowExclusive
  ) {
    if (numHeads <= 0 || numKvHeads <= 0 || headDim <= 0) {
      throw new IllegalArgumentException("numHeads, numKvHeads, and headDim must be > 0");
    }
    if (numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException("numHeads must be a multiple of numKvHeads");
    }
    if (windowExclusive < 0) {
      throw new IllegalArgumentException("windowExclusive must be >= 0");
    }
    this.numHeads = numHeads;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.repeats = numHeads / numKvHeads;
    this.scale = scale;
    this.windowExclusive = windowExclusive;
  }

  /**
   * @param query   {@code [seq, numHeads, headDim]}
   * @param key     {@code [seq, numKvHeads, headDim]}
   * @param value   {@code [seq, numKvHeads, headDim]}
   * @param context matmul pool, or {@code null} for the calling thread
   * @return {@code [seq, numHeads, headDim]}
   */
  public Tensor forward(
    final Tensor query,
    final Tensor key,
    final Tensor value,
    final Context context
  ) {
    requireNonNull(query, "query");
    requireNonNull(key, "key");
    requireNonNull(value, "value");
    int seq = query.size(0);
    if (key.size(0) != seq || value.size(0) != seq) {
      throw new IllegalArgumentException("query/key/value sequence lengths must match");
    }
    if (query.size(1) != this.numHeads || query.size(2) != this.headDim) {
      throw new IllegalArgumentException("query shape must be [seq, numHeads, headDim]");
    }
    if (key.size(1) != this.numKvHeads || value.size(1) != this.numKvHeads
      || key.size(2) != this.headDim || value.size(2) != this.headDim) {
      throw new IllegalArgumentException("key/value shape must be [seq, numKvHeads, headDim]");
    }

    Tensor out = Tensor.zeros(seq, this.numHeads, this.headDim);
    int work = this.numHeads * seq;
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    if (work * Math.max(seq, 1) < 256) {
      this.attendJobs(0, work, query, key, value, out, seq);
      return out;
    }
    runtime.parallelRanges(
      work, (start, end) -> this.attendJobs(start, end, query, key, value, out, seq));
    return out;
  }

  private void attendJobs(
    final int jobStart,
    final int jobEnd,
    final Tensor query,
    final Tensor key,
    final Tensor value,
    final Tensor out,
    final int seq
  ) {
    float[] queries = query.data();
    float[] keys = key.data();
    float[] values = value.data();
    float[] dest = out.data();
    int queryWidth = this.numHeads * this.headDim;
    int kvWidth = this.numKvHeads * this.headDim;
    int queryOff = query.offset();
    int keyOff = key.offset();
    int valueOff = value.offset();
    float[] scores = new float[seq];

    for (int job = jobStart; job < jobEnd; job++) {
      int head = job / seq;
      int token = job % seq;
      int kvHead = head / this.repeats;
      int queryAt = queryOff + token * queryWidth + head * this.headDim;
      int from = this.windowExclusive <= 0 ? 0 : Math.max(0, token - (this.windowExclusive - 1));
      int to = this.windowExclusive <= 0 ? seq : Math.min(seq, token + this.windowExclusive);
      float max = Float.NEGATIVE_INFINITY;
      for (int keyIndex = from; keyIndex < to; keyIndex++) {
        float score = VectorMath.dot(
          queries, queryAt,
          keys, keyOff + keyIndex * kvWidth + kvHead * this.headDim,
          this.headDim) * this.scale;
        scores[keyIndex] = score;
        if (score > max) {
          max = score;
        }
      }
      float sum = 0f;
      for (int keyIndex = from; keyIndex < to; keyIndex++) {
        float weight = (float) Math.exp(scores[keyIndex] - max);
        scores[keyIndex] = weight;
        sum += weight;
      }
      float inv = 1f / sum;
      int outAt = token * queryWidth + head * this.headDim;
      Arrays.fill(dest, outAt, outAt + this.headDim, 0f);
      for (int keyIndex = from; keyIndex < to; keyIndex++) {
        VectorMath.axpy(
          dest, outAt, scores[keyIndex] * inv,
          values, valueOff + keyIndex * kvWidth + kvHead * this.headDim,
          this.headDim);
      }
    }
  }
}
