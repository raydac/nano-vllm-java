package io.nanovllm.tensor;

import static java.util.Objects.requireNonNull;

public final class Ops {

  private Ops() {
  }

  public static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
    requireNonNull(x, "x");
    requireNonNull(weight, "weight");
    int[] xs = x.rawShape();
    int[] ws = weight.rawShape();
    if (ws.length != 2) {
      throw new IllegalArgumentException("weight must be 2D");
    }
    int in = ws[1];
    int out = ws[0];
    int rows = x.numel() / in;
    if (x.numel() % in != 0) {
      throw new IllegalArgumentException("x last dim mismatch");
    }
    Tensor y = Tensor.zeros(rows, out);
    float[] biasData = null;
    if (bias != null) {
      biasData = bias.offset() == 0 ? bias.data() : bias.toFloatArray();
    }
    VectorMath.linear(
        x.data(), x.offset(),
        weight.data(), weight.offset(),
        biasData,
        y.data(), 0,
        rows, in, out
    );
    if (xs.length == 1) {
      return y.reshape(out);
    }
    if (xs.length == 2) {
      return y.reshape(xs[0], out);
    }
    int[] newShape = xs.clone();
    newShape[newShape.length - 1] = out;
    return y.reshape(newShape);
  }

  public static Tensor embedding(Tensor ids, Tensor weight) {
    int[] ws = weight.rawShape();
    if (ws.length != 2) {
      throw new IllegalArgumentException("embedding weight must be [vocab, dim]");
    }
    int dim = ws[1];
    int n = ids.numel();
    Tensor out = Tensor.zeros(n, dim);
    for (int i = 0; i < n; i++) {
      int id = Math.round(ids.get(i));
      if (id < 0 || id >= ws[0]) {
        throw new IndexOutOfBoundsException("token id " + id);
      }
      System.arraycopy(weight.data(), weight.offset() + id * dim, out.data(), i * dim, dim);
    }
    return out;
  }

  public static Tensor siluAndMul(Tensor x) {
    return gatedActAndMul(x, false);
  }

  /**
   * Gemma MLP: gelu_pytorch_tanh(gate) * up.
   */
  public static Tensor geluPytorchTanhAndMul(Tensor x) {
    return gatedActAndMul(x, true);
  }

  private static Tensor gatedActAndMul(Tensor x, boolean geluTanh) {
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    if (last % 2 != 0) {
      throw new IllegalArgumentException("last dim must be even");
    }
    int half = last / 2;
    int rows = x.numel() / last;
    Tensor out = Tensor.zeros(rows, half);
    float[] xd = x.data();
    float[] od = out.data();
    int xOff = x.offset();
    for (int r = 0; r < rows; r++) {
      int base = xOff + r * last;
      int outBase = r * half;
      for (int i = 0; i < half; i++) {
        float gate = xd[base + i];
        float up = xd[base + half + i];
        float act = geluTanh ? geluPytorchTanh(gate) : (gate / (1.0f + (float) Math.exp(-gate)));
        od[outBase + i] = act * up;
      }
    }
    if (shape.length == 1) {
      return out.reshape(half);
    }
    int[] ns = shape.clone();
    ns[ns.length - 1] = half;
    return out.reshape(ns);
  }

  private static float geluPytorchTanh(float x) {
    return 0.5f * x * (1.0f + (float) Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }

  public static Tensor softmaxLastDim(Tensor logits) {
    int[] shape = logits.rawShape();
    int last = shape[shape.length - 1];
    int rows = logits.numel() / last;
    Tensor out = Tensor.zeros(logits.shape());
    float[] ld = logits.data();
    float[] od = out.data();
    int lOff = logits.offset();
    for (int r = 0; r < rows; r++) {
      int base = lOff + r * last;
      int outBase = r * last;
      float max = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < last; i++) {
        max = Math.max(max, ld[base + i]);
      }
      float sum = 0f;
      for (int i = 0; i < last; i++) {
        float e = (float) Math.exp(ld[base + i] - max);
        od[outBase + i] = e;
        sum += e;
      }
      float inv = 1.0f / sum;
      for (int i = 0; i < last; i++) {
        od[outBase + i] *= inv;
      }
    }
    return out;
  }

  public static Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
    return rmsNorm(x, weight, eps, false);
  }

  public static Tensor rmsNorm(Tensor x, Tensor weight, float eps, boolean gemmaStyle) {
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    int rows = x.numel() / last;
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] wd = weight.data();
    int xOff = x.offset();
    int wOff = weight.offset();
    float[] od = out.data();
    for (int r = 0; r < rows; r++) {
      int xBase = xOff + r * last;
      int oBase = r * last;
      float var = VectorMath.sumSquares(xd, xBase, last) / last;
      float inv = (float) (1.0 / Math.sqrt(var + eps));
      for (int i = 0; i < last; i++) {
        float w = wd[wOff + i];
        if (gemmaStyle) {
          w = 1.0f + w;
        }
        od[oBase + i] = xd[xBase + i] * inv * w;
      }
    }
    return out;
  }

  /**
   * Residual add + RMSNorm in one fused pass.
   * Returns {@code {normed, residualSum}} — same contract as before.
   */
  public static Tensor[] addRmsNorm(Tensor x, Tensor residual, Tensor weight, float eps) {
    return addRmsNorm(x, residual, weight, eps, false);
  }

  public static Tensor[] addRmsNorm(Tensor x, Tensor residual, Tensor weight, float eps,
                                    boolean gemmaStyle) {
    requireSameSize(x, residual);
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    int rows = x.numel() / last;
    Tensor summed = Tensor.zeros(x.shape());
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] rd = residual.data();
    float[] sd = summed.data();
    float[] od = out.data();
    float[] wd = weight.data();
    int xOff = x.offset();
    int rOff = residual.offset();
    int wOff = weight.offset();
    for (int r = 0; r < rows; r++) {
      int xBase = xOff + r * last;
      int rBase = rOff + r * last;
      int sBase = r * last;
      float sumSq = 0f;
      for (int i = 0; i < last; i++) {
        float v = xd[xBase + i] + rd[rBase + i];
        sd[sBase + i] = v;
        sumSq += v * v;
      }
      float inv = (float) (1.0 / Math.sqrt(sumSq / last + eps));
      for (int i = 0; i < last; i++) {
        float w = wd[wOff + i];
        if (gemmaStyle) {
          w = 1.0f + w;
        }
        od[sBase + i] = sd[sBase + i] * inv * w;
      }
    }
    return new Tensor[] {out, summed};
  }

  public static Tensor[] splitLast(Tensor x, int... sizes) {
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    int sum = 0;
    for (int s : sizes) {
      sum += s;
    }
    if (sum != last) {
      throw new IllegalArgumentException("split sizes must sum to last dim");
    }
    int rows = x.numel() / last;
    Tensor[] parts = new Tensor[sizes.length];
    int offset = 0;
    for (int p = 0; p < sizes.length; p++) {
      int sz = sizes[p];
      int[] ns = shape.clone();
      ns[ns.length - 1] = sz;
      Tensor part = Tensor.zeros(ns);
      for (int r = 0; r < rows; r++) {
        System.arraycopy(x.data(), x.offset() + r * last + offset, part.data(), r * sz, sz);
      }
      parts[p] = part;
      offset += sz;
    }
    return parts;
  }

  private static void requireSameSize(Tensor a, Tensor b) {
    if (a.numel() != b.numel()) {
      throw new IllegalArgumentException(
          "numel mismatch: %d vs %d".formatted(a.numel(), b.numel()));
    }
  }
}
