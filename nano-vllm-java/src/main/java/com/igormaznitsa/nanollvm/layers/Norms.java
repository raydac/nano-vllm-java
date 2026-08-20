package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.HashMap;
import java.util.Map;

/**
 * Normalization and rotary-embedding bricks used by architecture graphs.
 *
 * <p>{@link RMSNorm} is the chat default (optional {@code 1+w} gain, optional weightless identity
 * scale). {@link LayerNorm} is BERT. {@link RotaryEmbedding} applies precomputed cos/sin tables
 * to Q/K; {@link RotaryEmbedding.Tables} interns tables so layers that share a RoPE config share
 * one cache (owned by the model, not the process).
 */
public final class Norms {

  private Norms() {
  }

  /**
   * Immutable RMSNorm: scale vector is fixed at construction.
   *
   * <p>When {@code onePlusWeight} is true, stored {@code w} is applied as {@code (1 + w)} (Gemma 3
   * checkpoints store a delta from 1). {@link #weightless(float)} has no scale vector and uses
   * identity gain (Gemma 4 shared-V residual).
   */
  public static final class RMSNorm {
    private final float eps;
    private final boolean onePlusWeight;
    private final Tensor weight;

    /**
     * RMSNorm with stored weight applied as-is ({@code onePlusWeight == false}).
     *
     * @param weight per-channel scale, last-axis width
     * @param eps    stability epsilon
     */
    public RMSNorm(final Tensor weight, final float eps) {
      this(weight, eps, false);
    }

    /**
     * RMSNorm with an explicit {@code (1 + w)} vs stored-{@code w} rule.
     *
     * @param weight        per-channel scale
     * @param eps           stability epsilon
     * @param onePlusWeight {@code true} to apply {@code 1 + w} (Gemma 3)
     */
    public RMSNorm(final Tensor weight, final float eps, final boolean onePlusWeight) {
      this.weight = requireNonNull(weight, "weight");
      this.eps = eps;
      this.onePlusWeight = onePlusWeight;
    }

    private RMSNorm(final float eps) {
      this.weight = null;
      this.eps = eps;
      this.onePlusWeight = false;
    }

    /**
     * RMSNorm with identity scale ({@code weight == null}). Used where a residual still needs
     * RMS energy but the checkpoint has no gain tensor.
     *
     * @param eps stability epsilon
     * @return weightless norm
     * @since 1.1.0
     */
    public static RMSNorm weightless(final float eps) {
      return new RMSNorm(eps);
    }

    /**
     * Per-channel scale, or {@code null} for {@link #weightless(float)}.
     *
     * @return scale tensor, or {@code null}
     */
    public Tensor weight() {
      return this.weight;
    }

    /**
     * RMS-normalizes {@code x} along the last axis.
     *
     * @param x activations
     * @return normalized tensor, same shape as {@code x}
     */
    public Tensor forward(final Tensor x) {
      if (this.weight == null) {
        return Ops.rmsNorm(x, this.eps);
      }
      return Ops.rmsNorm(x, this.weight, this.eps, this.onePlusWeight);
    }

    /**
     * Fused residual add then RMSNorm: {@code summed = x + residual}, then this norm.
     *
     * @param x        branch output to add
     * @param residual incoming residual stream
     * @return {@code {normed, xPlusResidual}} — keep both; dropping the sum breaks the residual
     * @see Ops#addRmsNorm(Tensor, Tensor, Tensor, float, boolean)
     */
    public Tensor[] forward(final Tensor x, final Tensor residual) {
      if (this.weight == null) {
        return Ops.addRmsNorm(x, residual, this.eps);
      }
      return Ops.addRmsNorm(x, residual, this.weight, this.eps, this.onePlusWeight);
    }
  }

  /**
   * Immutable LayerNorm with affine weight and bias (BERT).
   *
   * @since 1.1.0
   */
  public static final class LayerNorm {
    private final float eps;
    private final Tensor weight;
    private final Tensor bias;

    /**
     * Affine LayerNorm along the last axis ({@code y = normalize(x) * weight + bias}).
     *
     * @param weight per-channel scale
     * @param bias   per-channel shift
     * @param eps    stability epsilon
     */
    public LayerNorm(final Tensor weight, final Tensor bias, final float eps) {
      this.weight = requireNonNull(weight, "weight");
      this.bias = requireNonNull(bias, "bias");
      this.eps = eps;
    }

    /**
     * Layer-normalizes {@code x} along the last axis, then affine {@code y = n * w + b}.
     *
     * @param x activations
     * @return normalized tensor, same shape as {@code x}
     */
    public Tensor forward(final Tensor x) {
      return Ops.layerNorm(x, this.weight, this.bias, this.eps);
    }
  }

  /**
   * Rotary position embedding: precomputed cos/sin tables applied to query and key heads.
   * Independent tokens may run on the step matmul pool when the rotate is large enough.
   *
   * <p>{@link #of} uses full-head inv-freq ({@code rotaryDim == headSize}).
   * {@link #proportional} leaves a tail of angles at inv-freq {@code 0} (identity rotation) for
   * Gemma 4 global RoPE ({@code partial_rotary_factor}).
   */
  public static final class RotaryEmbedding {
    private final int headSize;
    private final Tensor cosSinCache;

    /**
     * Full-head RoPE. {@code rotaryDim} must equal {@code headSize}.
     *
     * @param headSize              per-head width
     * @param rotaryDim             must equal {@code headSize}
     * @param maxPositionEmbeddings table length (positions {@code [0, max)})
     * @param base                  RoPE theta
     * @throws IllegalArgumentException if {@code rotaryDim != headSize}
     */
    public RotaryEmbedding(final int headSize, final int rotaryDim, final int maxPositionEmbeddings,
                           final float base) {
      this(headSize, maxPositionEmbeddings, defaultInvFreq(headSize, rotaryDim, base));
    }

    private RotaryEmbedding(final int headSize, final int maxPositionEmbeddings,
                            final float[] invFreq) {
      if (invFreq.length != headSize / 2) {
        throw new IllegalArgumentException("invFreq length must be headSize/2");
      }
      this.headSize = headSize;
      int half = headSize / 2;
      float[] cache = new float[maxPositionEmbeddings * headSize];
      for (int pos = 0; pos < maxPositionEmbeddings; pos++) {
        for (int i = 0; i < half; i++) {
          double freq = pos * invFreq[i];
          cache[pos * headSize + i] = (float) Math.cos(freq);
          cache[pos * headSize + half + i] = (float) Math.sin(freq);
        }
      }
      this.cosSinCache = Tensor.of(cache, maxPositionEmbeddings, headSize);
    }

    private static float[] defaultInvFreq(final int headSize, final int rotaryDim,
                                          final float base) {
      if (rotaryDim != headSize) {
        throw new IllegalArgumentException("rotaryDim must equal headSize");
      }
      int half = rotaryDim / 2;
      float[] invFreq = new float[half];
      for (int i = 0; i < half; i++) {
        invFreq[i] = (float) (1.0 / Math.pow(base, (2.0 * i) / rotaryDim));
      }
      return invFreq;
    }

    /**
     * Full-head RoPE table (same as the public constructor).
     *
     * @param headSize     per-head width
     * @param rotaryDim    must equal {@code headSize}
     * @param maxPosition  table length
     * @param base         RoPE theta
     * @return internable table
     */
    public static RotaryEmbedding of(
      final int headSize,
      final int rotaryDim,
      final int maxPosition,
      final float base
    ) {
      return new RotaryEmbedding(headSize, rotaryDim, maxPosition, base);
    }

    /**
     * Partial RoPE: only the first {@code partialRotaryFactor * headSize / 2} angle pairs rotate;
     * remaining pairs keep inv-freq {@code 0} (cos=1, sin=0). Gemma 4 global attention.
     *
     * @param headSize             per-head width
     * @param maxPosition          table length
     * @param base                 RoPE theta
     * @param partialRotaryFactor  fraction of the head that rotates, in {@code (0, 1]}
     * @return internable table
     * @since 1.1.0
     */
    public static RotaryEmbedding proportional(
      final int headSize,
      final int maxPosition,
      final float base,
      final float partialRotaryFactor
    ) {
      int half = headSize / 2;
      int ropeAngles = (int) (partialRotaryFactor * headSize / 2.0);
      float[] invFreq = new float[half];
      for (int i = 0; i < ropeAngles; i++) {
        invFreq[i] = (float) (1.0 / Math.pow(base, (2.0 * i) / headSize));
      }
      return new RotaryEmbedding(headSize, maxPosition, invFreq);
    }

    /**
     * Rotates {@code query} and {@code key} at the given token positions (sequential).
     *
     * @param positions rank-1 positions, one per token (rounded to int)
     * @param query     {@code [tokens, headsQ, headSize]}
     * @param key       {@code [tokens, headsK, headSize]}
     * @return {@code {rotatedQuery, rotatedKey}}
     * @throws IllegalArgumentException if a position is outside {@code [0, maxPosition)}
     */
    public Tensor[] forward(final Tensor positions, final Tensor query, final Tensor key) {
      return this.forward(positions, query, key, null);
    }

    /**
     * Rotates {@code query} and {@code key}. Independent tokens may run on
     * {@link Context#matmul()} when the work is large enough.
     *
     * @param positions rank-1 positions, one per token (rounded to int)
     * @param query     {@code [tokens, headsQ, headSize]}
     * @param key       {@code [tokens, headsK, headSize]}
     * @param context   step context (matmul pool), or {@code null} for sequential
     * @return {@code {rotatedQuery, rotatedKey}}
     * @throws IllegalArgumentException if a position is outside {@code [0, maxPosition)}
     */
    public Tensor[] forward(
      final Tensor positions,
      final Tensor query,
      final Tensor key,
      final Context context
    ) {
      int tokens = query.size(0);
      int headsQ = query.size(1);
      int headsK = key.size(1);
      Tensor qOut = Tensor.zeros(query.shape());
      Tensor kOut = Tensor.zeros(key.shape());
      int half = this.headSize / 2;
      int maxPos = this.cosSinCache.size(0);
      MatmulRuntime runtime = context != null && context.matmul() != null
        ? context.matmul()
        : MatmulRuntime.sequential();
      int cost = tokens * this.headSize * (headsQ + headsK);
      if (cost < 4096) {
        this.rotateTokens(
          0, tokens, positions, query, key, qOut, kOut, headsQ, headsK, half, maxPos);
        return new Tensor[] {qOut, kOut};
      }
      runtime.parallelRanges(
        tokens,
        (start, end) -> this.rotateTokens(
          start, end, positions, query, key, qOut, kOut, headsQ, headsK, half, maxPos));
      return new Tensor[] {qOut, kOut};
    }

    private void rotateTokens(
      final int tokenStart,
      final int tokenEnd,
      final Tensor positions,
      final Tensor query,
      final Tensor key,
      final Tensor qOut,
      final Tensor kOut,
      final int headsQ,
      final int headsK,
      final int half,
      final int maxPos
    ) {
      for (int t = tokenStart; t < tokenEnd; t++) {
        int pos = Math.round(positions.get(t));
        if (pos < 0 || pos >= maxPos) {
          throw new IllegalArgumentException(
            "RoPE position %d out of range [0, %d)".formatted(pos, maxPos));
        }
        int cBase = this.cosSinCache.offset() + pos * this.headSize;
        this.apply(query, qOut, t, headsQ, half, cBase);
        this.apply(key, kOut, t, headsK, half, cBase);
      }
    }

    /**
     * Applies the cos/sin pair at {@code cBase} to every head of one token.
     */
    private void apply(final Tensor in, final Tensor out, final int token, final int heads,
                       final int half, final int cBase) {
      for (int h = 0; h < heads; h++) {
        int base = in.offset() + (token * heads + h) * this.headSize;
        int oBase = (token * heads + h) * this.headSize;
        for (int i = 0; i < half; i++) {
          float x1 = in.data()[base + i];
          float x2 = in.data()[base + half + i];
          float cos = this.cosSinCache.data()[cBase + i];
          float sin = this.cosSinCache.data()[cBase + half + i];
          out.data()[oBase + i] = x1 * cos - x2 * sin;
          out.data()[oBase + half + i] = x2 * cos + x1 * sin;
        }
      }
    }

    /**
     * Interns RoPE tables by config key so layers that share head size / theta / window share one
     * cos/sin cache. Owned by the model graph, not process-wide.
     */
    public static final class Tables {
      private final Map<String, RotaryEmbedding> byKey = new HashMap<>();

      /**
       * Full-head table for {@code (headSize, rotaryDim, maxPosition, base)}, created once.
       *
       * @return cached {@link RotaryEmbedding#of}
       */
      public RotaryEmbedding get(
        final int headSize,
        final int rotaryDim,
        final int maxPosition,
        final float base
      ) {
        return this.byKey.computeIfAbsent(
          headSize + ":" + rotaryDim + ":" + maxPosition + ":" + base,
          k -> RotaryEmbedding.of(headSize, rotaryDim, maxPosition, base));
      }

      /**
       * Partial-RoPE table for Gemma 4 global layers, created once per key.
       *
       * @return cached {@link RotaryEmbedding#proportional}
       * @since 1.1.0
       */
      public RotaryEmbedding proportional(
        final int headSize,
        final int maxPosition,
        final float base,
        final float partialRotaryFactor
      ) {
        return this.byKey.computeIfAbsent(
          "p:" + headSize + ":" + maxPosition + ":" + base + ":" + partialRotaryFactor,
          k -> RotaryEmbedding.proportional(headSize, maxPosition, base, partialRotaryFactor));
      }
    }
  }
}
