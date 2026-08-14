package com.igormaznitsa.nanollvm.tensor;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import java.util.Arrays;

/**
 * High-level tensor operations used by transformer layers (linear, embedding, norms, MLP gates,
 * softmax, splits).
 *
 * <h2>What this class is</h2>
 * {@link Tensor} owns storage and shape; {@link FloatKernels} / {@link VectorMath} own raw float
 * loops and {@link MatmulRuntime} owns parallel dense matmul. {@code Ops} sits between them: it
 * interprets {@link Tensor} layouts, allocates outputs, and implements the algebraic bricks that
 * {@code Linear}, {@code RMSNorm}, embeddings, and MLPs call.
 *
 * <p>All public methods are <strong>static</strong> and side-effect-free on inputs (they allocate
 * new result tensors unless noted). Inputs may be views with non-zero {@link Tensor#offset()}.
 *
 * <h2>Layout conventions (hard parts)</h2>
 * <ul>
 *   <li><strong>Last-axis semantics:</strong> many ops treat the last dimension as the feature
 *       width {@code H} (or vocabulary / intermediate width) and pack leading axes as “rows”
 *       ({@code numel / last}).</li>
 *   <li><strong>Linear weights:</strong> shape {@code [out, in]} — each output channel is a row of
 *       length {@code in}, matching Hugging Face / this port’s load layout.</li>
 *   <li><strong>Embedding weights:</strong> shape {@code [vocab, dim]} — row {@code id} is the
 *       vector for token {@code id} (gather, not a matmul).</li>
 *   <li><strong>Gated MLP:</strong> last dim is {@code 2 * half}; first half = gate, second = up.</li>
 *   <li><strong>Offset RMSNorm:</strong> when {@code onePlusWeight}, stored weight {@code w} is applied
 *       as {@code (1 + w)} (some HF checkpoints store a delta from 1).</li>
 * </ul>
 *
 * @see Tensor
 * @see MatmulRuntime
 * @see VectorMath
 * @see FloatKernels
 */
public final class Ops {

  private Ops() {
  }

  /**
   * Affine map along the last axis: {@code y = x @ Wᵀ (+ bias)} in row-major layout terms.
   *
   * <p><strong>Shapes:</strong> {@code weight} must be {@code [out, in]}. {@code x}’s number of
   * elements must be a multiple of {@code in}; it is viewed as {@code [rows, in]} where
   * {@code rows = x.numel() / in}. Output is {@code [rows, out]}, then reshaped to preserve
   * {@code x}’s leading axes with the last axis replaced by {@code out} (rank-1 {@code x} becomes
   * a length-{@code out} vector).
   *
   * <p><strong>Hard part — bias buffer:</strong> if {@code bias} is a non-zero-offset view,
   * elements are copied via {@link Tensor#toFloatArray()} so {@link MatmulRuntime#linear} can index
   * bias from {@code 0}. Weight and activation slices keep their offsets.
   *
   * @param x      input activations (last logical width = {@code in})
   * @param weight matrix {@code [out, in]}
   * @param bias   length-{@code out} bias, or {@code null} for none
   * @return transformed tensor with last dim {@code out}
   * @throws IllegalArgumentException if {@code weight} is not 2D or {@code x} width mismatches
   */
  public static Tensor linear(final Tensor x, final Tensor weight, final Tensor bias) {
    return linear(x, weight, bias, MatmulRuntime.sequential());
  }

  /**
   * Same as {@link #linear(Tensor, Tensor, Tensor)} using the matmul runtime bound on
   * {@code context}, or {@link MatmulRuntime#sequential()} when unbound / {@code null}.
   */
  public static Tensor linear(
    final Tensor x,
    final Tensor weight,
    final Tensor bias,
    final Context context) {
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    return linear(x, weight, bias, runtime);
  }

  public static Tensor linear(
    final Tensor x,
    final Tensor weight,
    final Tensor bias,
    final MatmulRuntime matmul) {
    requireNonNull(x, "x");
    requireNonNull(weight, "weight");
    requireNonNull(matmul, "matmul");
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
    matmul.linear(
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

  /**
   * Affine map with a packed GGUF weight {@code [out, in]}: dequantizes one weight row at a time.
   */
  public static Tensor linear(
    final Tensor x,
    final PackedWeight weight,
    final Tensor bias,
    final Context context) {
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    return linear(x, weight, bias, runtime);
  }

  public static Tensor linear(
    final Tensor x,
    final PackedWeight weight,
    final Tensor bias,
    final MatmulRuntime matmul) {
    requireNonNull(x, "x");
    requireNonNull(weight, "weight");
    requireNonNull(matmul, "matmul");
    LinearKernel kernel = LinearKernel.of(weight);
    int[] xs = x.rawShape();
    int in = kernel.inFeatures();
    int out = kernel.outFeatures();
    int rows = x.numel() / in;
    if (x.numel() % in != 0) {
      throw new IllegalArgumentException("x last dim mismatch");
    }
    Tensor y = Tensor.zeros(rows, out);
    float[] biasData = null;
    if (bias != null) {
      biasData = bias.offset() == 0 ? bias.data() : bias.toFloatArray();
    }
    kernel.apply(x.data(), x.offset(), biasData, y.data(), 0, rows, matmul);
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

  /**
   * Token embedding lookup: copy row {@code id} from {@code weight} for each id in {@code ids}.
   *
   * <p>Not a matrix multiply — a <strong>gather</strong>. {@code weight} is {@code [vocab, dim]};
   * each id is {@link Math#round(float) rounded} from the float storage of {@code ids} (token ids
   * are carried in float tensors elsewhere in the engine).
   *
   * @param ids    flat (or multi-dim) tensor of token ids; {@code numel} = batch of lookups
   * @param weight embedding table {@code [vocab_size, hidden_size]}
   * @return {@code [numel(ids), dim]} of concatenated rows
   * @throws IllegalArgumentException  if {@code weight} is not {@code [vocab, dim]}
   * @throws IndexOutOfBoundsException if any id is outside {@code [0, vocab)}
   */
  public static Tensor embedding(final Tensor ids, final Tensor weight) {
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

  /**
   * Embedding gather from a packed GGUF table {@code [vocab, dim]} (dequant one row per id).
   */
  public static Tensor embedding(final Tensor ids, final PackedWeight weight) {
    requireNonNull(ids, "ids");
    requireNonNull(weight, "weight");
    EmbeddingKernel kernel = EmbeddingKernel.of(weight);
    int n = ids.numel();
    Tensor out = Tensor.zeros(n, kernel.embeddingDim());
    kernel.gather(ids.data(), ids.offset(), n, out.data(), 0);
    return out;
  }

  /**
   * SwiGLU-style gate: {@code silu(gate) * up} with gate/up packed in the last dimension.
   *
   * <p>SiLU (swish) is {@code x / (1 + exp(-x))}. See {@link #gatedActAndMul(Tensor, boolean)}.
   *
   * @param x last dim even; layout {@code […, gate | up]}
   * @return tensor with last dim halved
   */
  public static Tensor siluAndMul(final Tensor x) {
    return gatedActAndMul(x, false);
  }

  /**
   * SwiGLU with separate gate and up tensors: {@code silu(gate) * up} (unfused MLP).
   */
  public static Tensor siluAndMul(final Tensor gate, final Tensor up) {
    requireSameShape(gate, up, "gate", "up");
    Tensor out = Tensor.zeros(gate.shape());
    float[] gd = gate.data();
    float[] ud = up.data();
    float[] od = out.data();
    int gOff = gate.offset();
    int uOff = up.offset();
    int n = gate.numel();
    for (int i = 0; i < n; i++) {
      float g = gd[gOff + i];
      od[i] = (g / (1.0f + (float) Math.exp(-g))) * ud[uOff + i];
    }
    return out;
  }

  /**
   * Elementwise sum of two same-shaped tensors (residual add).
   */
  public static Tensor add(final Tensor a, final Tensor b) {
    requireSameShape(a, b, "a", "b");
    Tensor out = Tensor.zeros(a.shape());
    float[] ad = a.data();
    float[] bd = b.data();
    float[] od = out.data();
    int aOff = a.offset();
    int bOff = b.offset();
    int n = a.numel();
    for (int i = 0; i < n; i++) {
      od[i] = ad[aOff + i] + bd[bOff + i];
    }
    return out;
  }

  public static Tensor mul(final Tensor a, final Tensor b) {
    requireSameShape(a, b, "a", "b");
    Tensor out = Tensor.zeros(a.shape());
    float[] ad = a.data();
    float[] bd = b.data();
    float[] od = out.data();
    int aOff = a.offset();
    int bOff = b.offset();
    int n = a.numel();
    for (int i = 0; i < n; i++) {
      od[i] = ad[aOff + i] * bd[bOff + i];
    }
    return out;
  }

  public static Tensor scale(final Tensor x, final float factor) {
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] od = out.data();
    int off = x.offset();
    int n = x.numel();
    for (int i = 0; i < n; i++) {
      od[i] = xd[off + i] * factor;
    }
    return out;
  }

  public static Tensor tanhSoftcap(final Tensor logits, final float cap) {
    if (cap <= 0f) {
      return logits;
    }
    Tensor out = Tensor.zeros(logits.shape());
    float[] ld = logits.data();
    float[] od = out.data();
    int off = logits.offset();
    int n = logits.numel();
    for (int i = 0; i < n; i++) {
      od[i] = (float) Math.tanh(ld[off + i] / cap) * cap;
    }
    return out;
  }

  /**
   * PyTorch-style GELU approximate with tanh (used by BERT / some causal MLP gates).
   *
   * @since 1.1.0
   */
  public static Tensor gelu(final Tensor x) {
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] od = out.data();
    int xOff = x.offset();
    int n = x.numel();
    for (int i = 0; i < n; i++) {
      od[i] = geluPytorchTanh(xd[xOff + i]);
    }
    return out;
  }

  /**
   * LayerNorm along the last axis: {@code (x - mean) / sqrt(var + eps) * weight + bias}.
   *
   * @since 1.1.0
   */
  public static Tensor layerNorm(
    final Tensor x,
    final Tensor weight,
    final Tensor bias,
    final float eps
  ) {
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    int rows = x.numel() / last;
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] wd = weight.data();
    float[] bd = bias.data();
    float[] od = out.data();
    int xOff = x.offset();
    int wOff = weight.offset();
    int bOff = bias.offset();
    for (int r = 0; r < rows; r++) {
      int xBase = xOff + r * last;
      int oBase = r * last;
      float sum = 0f;
      for (int i = 0; i < last; i++) {
        sum += xd[xBase + i];
      }
      float mean = sum / last;
      float var = 0f;
      for (int i = 0; i < last; i++) {
        float d = xd[xBase + i] - mean;
        var += d * d;
      }
      float inv = (float) (1.0 / Math.sqrt(var / last + eps));
      for (int i = 0; i < last; i++) {
        od[oBase + i] = (xd[xBase + i] - mean) * inv * wd[wOff + i] + bd[bOff + i];
      }
    }
    return out;
  }

  private static void requireSameShape(
    final Tensor left,
    final Tensor right,
    final String leftName,
    final String rightName
  ) {
    if (!Arrays.equals(left.rawShape(), right.rawShape())) {
      throw new IllegalArgumentException(
        leftName + " shape " + Arrays.toString(left.rawShape())
          + " != " + rightName + " shape " + Arrays.toString(right.rawShape()));
    }
  }

  /**
   * MLP gate: {@code gelu_pytorch_tanh(gate) * up} with gate/up packed in the last dimension.
   *
   * <p>Uses the tanh approximation of GELU ({@code gelu_pytorch_tanh}), not {@code erf}.
   * See {@link #gatedActAndMul(Tensor, boolean)} and {@link #geluPytorchTanh(float)}.
   *
   * @param x last dim even; layout {@code […, gate | up]}
   * @return tensor with last dim halved
   */
  public static Tensor geluPytorchTanhAndMul(final Tensor x) {
    return gatedActAndMul(x, true);
  }

  /**
   * Shared gated-MLP body: split last axis into gate (first half) and up (second half), activate
   * gate, multiply by up elementwise.
   *
   * <p><strong>Hard part — packing:</strong> fused {@code gate_up_proj} outputs
   * {@code […, 2*half]}; this op avoids an explicit {@link #splitLast} + two tensors by indexing
   * {@code base + i} (gate) and {@code base + half + i} (up) in one pass. Output last dim is
   * {@code half}, leading axes preserved.
   *
   * @param x        activations with even last dimension
   * @param geluTanh {@code true} → GELU-tanh; {@code false} → SiLU
   * @return activated-and-multiplied tensor
   * @throws IllegalArgumentException if last dim is odd
   */
  private static Tensor gatedActAndMul(final Tensor x, final boolean geluTanh) {
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

  /**
   * PyTorch {@code gelu} approximate with tanh:
   * {@code 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))}.
   *
   * <p>Constant {@code 0.7978845608028654} is {@code √(2/π)}.
   */
  private static float geluPytorchTanh(final float x) {
    return 0.5f * x * (1.0f + (float) Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }

  /**
   * Softmax independently along the <strong>last</strong> axis of each row.
   *
   * <p><strong>Hard part — stability:</strong> subtracts the row max before {@code exp}, then
   * normalizes so each row sums to 1. Used for attention weights and related distributions.
   *
   * @param logits scores with arbitrary leading shape and last dim = class / key count
   * @return same shape as {@code logits}; non-negative, last-axis sums to ~1
   */
  public static Tensor softmaxLastDim(final Tensor logits) {
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

  /**
   * RMSNorm with {@code onePlusWeight == false} (weight applied as stored).
   *
   * @see #rmsNorm(Tensor, Tensor, float, boolean)
   */
  public static Tensor rmsNorm(final Tensor x, final Tensor weight, final float eps) {
    return rmsNorm(x, weight, eps, false);
  }

  public static Tensor rmsNorm(final Tensor x, final float eps) {
    int[] shape = x.rawShape();
    int last = shape[shape.length - 1];
    int rows = x.numel() / last;
    Tensor out = Tensor.zeros(x.shape());
    float[] xd = x.data();
    int xOff = x.offset();
    float[] od = out.data();
    for (int r = 0; r < rows; r++) {
      int xBase = xOff + r * last;
      int oBase = r * last;
      float var = VectorMath.sumSquares(xd, xBase, last) / last;
      float inv = (float) (1.0 / Math.sqrt(var + eps));
      for (int i = 0; i < last; i++) {
        od[oBase + i] = xd[xBase + i] * inv;
      }
    }
    return out;
  }

  /**
   * Root-mean-square layer norm along the last axis.
   *
   * <p>For each row of length {@code H}: {@code inv = 1 / sqrt(mean(x²) + eps)}, then
   * {@code out = x * inv * w'}. Uses {@link VectorMath#sumSquares} for the energy.
   *
   * <p>If {@code onePlusWeight}, {@code w' = 1 + weight[i]} (checkpoint stores a delta from 1).
   * Otherwise {@code w' = weight[i]}.
   *
   * @param x          input; last dim = feature width
   * @param weight     length-{@code H} scale vector
   * @param eps        added under the square root for stability
   * @param onePlusWeight whether to use {@code (1 + w)} scales
   * @return same shape as {@code x}
   */
  public static Tensor rmsNorm(final Tensor x, final Tensor weight, final float eps,
                               final boolean onePlusWeight) {
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
        if (onePlusWeight) {
          w = 1.0f + w;
        }
        od[oBase + i] = xd[xBase + i] * inv * w;
      }
    }
    return out;
  }

  /**
   * Residual add + RMSNorm ({@code onePlusWeight == false}).
   *
   * @see #addRmsNorm(Tensor, Tensor, Tensor, float, boolean)
   */
  public static Tensor[] addRmsNorm(final Tensor x, final Tensor residual, final Tensor weight,
                                    final float eps) {
    return addRmsNorm(x, residual, weight, eps, false);
  }

  /**
   * Fused residual add and RMSNorm: {@code summed = x + residual}, then RMSNorm({@code summed}).
   *
   * <p><strong>Hard part — return contract:</strong> returns {@code {normed, residualSum}} where
   * index {@code 0} is the normalized tensor (fed to the next sublayer) and index {@code 1} is the
   * post-add residual stream to carry forward. Callers must keep both; dropping {@code summed}
   * breaks the residual highway. Same {@code onePlusWeight} weight rule as {@link #rmsNorm}.
   *
   * @param x          branch output to add
   * @param residual   incoming residual stream (same {@link Tensor#numel()} as {@code x})
   * @param weight     RMSNorm scale
   * @param eps        stability epsilon
   * @param onePlusWeight {@code (1 + w)} if true
   * @return {@code new Tensor[] { normed, xPlusResidual }}
   * @throws IllegalArgumentException if {@code x} and {@code residual} sizes differ
   */
  public static Tensor[] addRmsNorm(final Tensor x, final Tensor residual, final Tensor weight,
                                    final float eps,
                                    final boolean onePlusWeight) {
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
        if (onePlusWeight) {
          w = 1.0f + w;
        }
        od[sBase + i] = sd[sBase + i] * inv * w;
      }
    }
    return new Tensor[] {out, summed};
  }

  /**
   * Split the last axis into contiguous chunks of the given sizes.
   *
   * <p>Used when Q/K/V (or similar) are packed in one tensor and must become separate tensors.
   * Each part is a dense copy (not a view). Leading axes are preserved; only the last dim changes
   * to each chunk size.
   *
   * @param x     source; last dim must equal {@code sum(sizes)}
   * @param sizes positive chunk widths along the last axis
   * @return one tensor per size, in order
   * @throws IllegalArgumentException if sizes do not sum to the last dimension
   */
  public static Tensor[] splitLast(final Tensor x, int... sizes) {
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

  /**
   * Guards fused residual ops: both views must expose the same number of scalars.
   */
  private static void requireSameSize(final Tensor a, final Tensor b) {
    if (a.numel() != b.numel()) {
      throw new IllegalArgumentException(
        "numel mismatch: %d vs %d".formatted(a.numel(), b.numel()));
    }
  }
}
