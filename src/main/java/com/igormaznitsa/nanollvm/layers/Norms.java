package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public final class Norms {

  private Norms() {
  }

  /**
   * Immutable RMSNorm: scale vector is fixed at construction.
   */
  public static final class RMSNorm {
    private final float eps;
    private final boolean gemmaStyle;
    private final Tensor weight;

    public RMSNorm(final Tensor weight, final float eps) {
      this(weight, eps, false);
    }

    public RMSNorm(final Tensor weight, final float eps, final boolean gemmaStyle) {
      this.weight = requireNonNull(weight, "weight");
      this.eps = eps;
      this.gemmaStyle = gemmaStyle;
    }

    public Tensor weight() {
      return this.weight;
    }

    public Tensor forward(final Tensor x) {
      return Ops.rmsNorm(x, this.weight, this.eps, this.gemmaStyle);
    }

    public Tensor[] forward(final Tensor x, final Tensor residual) {
      return Ops.addRmsNorm(x, residual, this.weight, this.eps, this.gemmaStyle);
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

    public LayerNorm(final Tensor weight, final Tensor bias, final float eps) {
      this.weight = requireNonNull(weight, "weight");
      this.bias = requireNonNull(bias, "bias");
      this.eps = eps;
    }

    public Tensor forward(final Tensor x) {
      return Ops.layerNorm(x, this.weight, this.bias, this.eps);
    }
  }

  public static final class RotaryEmbedding {
    private static final Map<String, RotaryEmbedding> CACHE = new HashMap<>();

    private final int headSize;
    private final Tensor cosSinCache;

    public RotaryEmbedding(final int headSize, final int rotaryDim, final int maxPositionEmbeddings,
                           final float base) {
      if (rotaryDim != headSize) {
        throw new IllegalArgumentException("rotaryDim must equal headSize");
      }
      this.headSize = headSize;
      int half = rotaryDim / 2;
      float[] invFreq = new float[half];
      for (int i = 0; i < half; i++) {
        invFreq[i] = (float) (1.0 / Math.pow(base, (2.0 * i) / rotaryDim));
      }
      float[] cache = new float[maxPositionEmbeddings * rotaryDim];
      for (int pos = 0; pos < maxPositionEmbeddings; pos++) {
        for (int i = 0; i < half; i++) {
          double freq = pos * invFreq[i];
          cache[pos * rotaryDim + i] = (float) Math.cos(freq);
          cache[pos * rotaryDim + half + i] = (float) Math.sin(freq);
        }
      }
      this.cosSinCache = Tensor.of(cache, maxPositionEmbeddings, rotaryDim);
    }

    public static RotaryEmbedding get(final int headSize, final int rotaryDim,
                                      final int maxPosition, final float base) {
      String key = headSize + ":" + rotaryDim + ":" + maxPosition + ":" + base;
      return CACHE.computeIfAbsent(key,
          k -> new RotaryEmbedding(headSize, rotaryDim, maxPosition, base));
    }

    public Tensor[] forward(final Tensor positions, final Tensor query, final Tensor key) {
      int tokens = query.size(0);
      int headsQ = query.size(1);
      int headsK = key.size(1);
      Tensor qOut = Tensor.zeros(query.shape());
      Tensor kOut = Tensor.zeros(key.shape());
      int half = this.headSize / 2;
      for (int t = 0; t < tokens; t++) {
        int pos = Math.round(positions.get(t));
        int cBase = this.cosSinCache.offset() + pos * this.headSize;
        this.apply(query, qOut, t, headsQ, half, cBase);
        this.apply(key, kOut, t, headsK, half, cBase);
      }
      return new Tensor[] {qOut, kOut};
    }

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
  }
}
