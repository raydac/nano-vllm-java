package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.engine.ConvStateArena;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Gated short convolution: {@code in_proj → B,C,x → conv1d(B*x) → C*y → out_proj}.
 * LFM2 hybrid layers use this instead of attention on conv blocks.
 *
 * <p>{@code in_proj} emits {@code 3 * hidden} channels (gate B, gate C, value x). The depthwise
 * conv is causal with kernel width {@code >= 2}. Prefill writes the last {@code kernelSize - 1}
 * tokens into {@link ConvStateArena}; decode consumes that state one token at a time. When no
 * arena is bound, the conv still runs but state is not kept (stateless prefill).
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; used on the generate thread only.
 */
public final class ShortConv {

  private final Linear.Row inProj;
  private final Linear.Row outProj;
  private final float[] kernel;
  private final int hiddenSize;
  private final int kernelSize;
  private final int layerIndex;

  /**
   * Dense float32 short-conv (HF / unpacked GGUF).
   *
   * @param inProjWeight  {@code [3*hidden, hidden]}
   * @param convWeight    depthwise kernel {@code [hidden, kernelSize]}
   * @param outProjWeight {@code [hidden, hidden]}
   * @param layerIndex    conv-arena layer this block writes
   * @throws IllegalArgumentException if {@code kernelSize < 2}
   */
  public ShortConv(
    final Tensor inProjWeight,
    final Tensor convWeight,
    final Tensor outProjWeight,
    final int layerIndex
  ) {
    this(
      new Linear.Row(inProjWeight),
      convWeight,
      new Linear.Row(outProjWeight),
      layerIndex);
  }

  /**
   * Packed in/out projections with a dense conv kernel (GGUF LFM2 default).
   *
   * @param inProjWeight  packed {@code [3*hidden, hidden]}
   * @param convWeight    depthwise kernel {@code [hidden, kernelSize]} (always dense)
   * @param outProjWeight packed {@code [hidden, hidden]}
   * @param layerIndex    conv-arena layer this block writes
   * @throws IllegalArgumentException if {@code kernelSize < 2}
   */
  public ShortConv(
    final PackedWeight inProjWeight,
    final Tensor convWeight,
    final PackedWeight outProjWeight,
    final int layerIndex
  ) {
    this(
      new Linear.Row(inProjWeight),
      convWeight,
      new Linear.Row(outProjWeight),
      layerIndex);
  }

  private ShortConv(
    final Linear.Row inProj,
    final Tensor convWeight,
    final Linear.Row outProj,
    final int layerIndex
  ) {
    this.inProj = requireNonNull(inProj, "inProj");
    this.outProj = requireNonNull(outProj, "outProj");
    requireNonNull(convWeight, "convWeight");
    this.hiddenSize = convWeight.size(0);
    this.kernelSize = convWeight.size(1);
    if (this.kernelSize < 2) {
      throw new IllegalArgumentException("shortconv kernel size must be >= 2");
    }
    this.kernel = convWeight.toFloatArray();
    this.layerIndex = layerIndex;
  }

  /**
   * One short-conv pass over a scheduled batch. Requires {@code ctx} (matmul + optional conv
   * arena / seq ids). Prefill uses {@link Context#cuSeqlensQ()}; decode uses one token per row.
   *
   * @param hiddenStates {@code [tokens, hidden]}
   * @param ctx          step context
   * @return {@code [tokens, hidden]}
   * @throws NullPointerException  if {@code ctx} is {@code null}
   * @throws IllegalStateException if {@code in_proj} width is not {@code 3 * hidden}
   */
  public Tensor forward(final Tensor hiddenStates, final Context ctx) {
    requireNonNull(ctx, "ctx");
    Tensor projected = this.inProj.forward(hiddenStates, ctx);
    int tokens = projected.size(0);
    int width = projected.size(1);
    if (width != 3 * this.hiddenSize) {
      throw new IllegalStateException(
        "shortconv in_proj width " + width + " != 3*" + this.hiddenSize);
    }

    float[] pd = projected.data();
    int pOff = projected.offset();
    float[] bx = new float[tokens * this.hiddenSize];
    float[] gateC = new float[tokens * this.hiddenSize];
    for (int t = 0; t < tokens; t++) {
      int base = pOff + t * width;
      int row = t * this.hiddenSize;
      for (int h = 0; h < this.hiddenSize; h++) {
        float b = pd[base + h];
        gateC[row + h] = pd[base + this.hiddenSize + h];
        float x = pd[base + 2 * this.hiddenSize + h];
        bx[row + h] = b * x;
      }
    }

    float[] convOut = this.causalConv(bx, tokens, ctx);
    for (int i = 0; i < convOut.length; i++) {
      convOut[i] *= gateC[i];
    }
    return this.outProj.forward(Tensor.of(convOut, tokens, this.hiddenSize), ctx);
  }

  /**
   * Causal depthwise conv of {@code B*x}. Prefill writes trailing window into the arena;
   * decode reads/updates one token. No arena → stateless range conv.
   */
  private float[] causalConv(final float[] bx, final int tokens, final Context ctx) {
    ConvStateArena arena = ctx.convCache();
    int[] seqIds = ctx.seqIds();
    int stateLen = this.kernelSize - 1;
    float[] out = new float[tokens * this.hiddenSize];

    if (arena == null || seqIds == null) {
      this.prefillRange(bx, 0, tokens, out, null, stateLen);
      return out;
    }

    if (ctx.isPrefill()) {
      int[] cuQ = ctx.cuSeqlensQ();
      for (int b = 0; b < cuQ.length - 1; b++) {
        int start = cuQ[b];
        int end = cuQ[b + 1];
        if (end <= start) {
          continue;
        }
        int seqId = seqIds[start];
        float[] state = arena.state(seqId, this.layerIndex);
        this.prefillRange(bx, start, end, out, state, stateLen);
      }
      return out;
    }

    for (int t = 0; t < tokens; t++) {
      float[] state = arena.state(seqIds[t], this.layerIndex);
      this.decodeStep(bx, t, out, state, stateLen);
    }
    return out;
  }

  /**
   * Convolves tokens {@code [start, end)} with left-padding from {@code stateOut} (or zeros).
   * When {@code stateOut} is non-null, writes the trailing {@code stateLen} tokens back.
   */
  private void prefillRange(
    final float[] bx,
    final int start,
    final int end,
    final float[] out,
    final float[] stateOut,
    final int stateLen
  ) {
    int seqTokens = end - start;
    int padded = stateLen + seqTokens;
    float[] window = new float[padded * this.hiddenSize];
    if (stateOut != null) {
      System.arraycopy(stateOut, 0, window, 0,
        Math.min(stateOut.length, stateLen * this.hiddenSize));
    }
    for (int t = 0; t < seqTokens; t++) {
      System.arraycopy(
        bx, (start + t) * this.hiddenSize,
        window, (stateLen + t) * this.hiddenSize,
        this.hiddenSize);
    }
    for (int t = 0; t < seqTokens; t++) {
      this.convAt(window, stateLen + t, out, (start + t) * this.hiddenSize);
    }
    if (stateOut != null) {
      System.arraycopy(
        window, seqTokens * this.hiddenSize,
        stateOut, 0,
        stateLen * this.hiddenSize);
    }
  }

  /**
   * One decode token: pad with arena state, conv, then shift the window back into {@code state}.
   */
  private void decodeStep(
    final float[] bx,
    final int tokenIndex,
    final float[] out,
    final float[] state,
    final int stateLen
  ) {
    float[] window = new float[(stateLen + 1) * this.hiddenSize];
    System.arraycopy(state, 0, window, 0, state.length);
    System.arraycopy(bx, tokenIndex * this.hiddenSize, window, stateLen * this.hiddenSize,
      this.hiddenSize);
    this.convAt(window, stateLen, out, tokenIndex * this.hiddenSize);
    System.arraycopy(window, this.hiddenSize, state, 0, stateLen * this.hiddenSize);
  }

  /**
   * Depthwise dot of {@code kernelSize} window steps ending at {@code endIndexInclusive}.
   */
  private void convAt(
    final float[] window,
    final int endIndexInclusive,
    final float[] out,
    final int outBase
  ) {
    int start = endIndexInclusive - this.kernelSize + 1;
    for (int h = 0; h < this.hiddenSize; h++) {
      float sum = 0f;
      for (int k = 0; k < this.kernelSize; k++) {
        int t = start + k;
        sum += window[t * this.hiddenSize + h] * this.kernel[h * this.kernelSize + k];
      }
      out[outBase + h] = sum;
    }
  }
}
