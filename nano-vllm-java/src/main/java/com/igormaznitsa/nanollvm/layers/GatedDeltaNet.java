package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.engine.DeltaNetStateArena;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Qwen3.5 linear-attention block: causal depthwise conv, then a gated delta-rule recurrence,
 * then RMSNorm gated by SiLU. Full-attention layers in the same stack do not use this class.
 *
 * <p>Conv state is the last {@code kernel - 1} pre-activation channels. The recurrent state is
 * {@code [valueHeads, keyHeadDim, valueHeadDim]} and is updated in place on the bound
 * {@link DeltaNetStateArena}. A missing arena still runs the step from zeros and drops the result.
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; generate thread only.
 *
 * @since 1.5.0
 */
public final class GatedDeltaNet {

  private final Linear.Row qkvProj;
  private final Linear.Row zProj;
  private final Linear.Row betaProj;
  private final Linear.Row alphaProj;
  private final Linear.Row outProj;
  private final float[] convKernel;
  private final float[] aLog;
  private final float[] dtBias;
  private final Tensor normWeight;
  private final int convDim;
  private final int kernelSize;
  private final int keyDim;
  private final int valueDim;
  private final int numKeyHeads;
  private final int numValueHeads;
  private final int keyHeadDim;
  private final int valueHeadDim;
  private final int headGroup;
  private final float queryScale;
  private final float normEps;
  private final int layerIndex;

  /**
   * @param qkvWeight     {@code [2 * keyDim + valueDim, hidden]}
   * @param zWeight       output gate {@code [valueDim, hidden]}
   * @param betaWeight    {@code [numValueHeads, hidden]}
   * @param alphaWeight   {@code [numValueHeads, hidden]}
   * @param convWeight    depthwise kernel {@code [convDim, kernel]} or {@code [convDim, 1, kernel]}
   * @param aLog          per-value-head log decay, length {@code numValueHeads}
   * @param dtBias        per-value-head bias added before softplus
   * @param normWeight    gated RMSNorm scale, length {@code valueHeadDim}
   * @param outWeight     {@code [hidden, valueDim]}
   * @param numKeyHeads   linear key heads
   * @param keyHeadDim    key head width
   * @param numValueHeads linear value heads (multiple of {@code numKeyHeads})
   * @param valueHeadDim  value head width
   * @param normEps       RMSNorm epsilon
   * @param layerIndex    delta-state arena layer
   */
  public GatedDeltaNet(
    final Tensor qkvWeight,
    final Tensor zWeight,
    final Tensor betaWeight,
    final Tensor alphaWeight,
    final Tensor convWeight,
    final Tensor aLog,
    final Tensor dtBias,
    final Tensor normWeight,
    final Tensor outWeight,
    final int numKeyHeads,
    final int keyHeadDim,
    final int numValueHeads,
    final int valueHeadDim,
    final float normEps,
    final int layerIndex
  ) {
    this.qkvProj = new Linear.Row(requireNonNull(qkvWeight, "qkvWeight"));
    this.zProj = new Linear.Row(requireNonNull(zWeight, "zWeight"));
    this.betaProj = new Linear.Row(requireNonNull(betaWeight, "betaWeight"));
    this.alphaProj = new Linear.Row(requireNonNull(alphaWeight, "alphaWeight"));
    this.outProj = new Linear.Row(requireNonNull(outWeight, "outWeight"));
    this.convKernel = kernelOf(requireNonNull(convWeight, "convWeight"));
    this.kernelSize = convWeight.ndim() == 3 ? convWeight.size(2) : convWeight.size(1);
    this.convDim = convWeight.size(0);
    if (this.kernelSize < 2) {
      throw new IllegalArgumentException("conv kernel must be >= 2");
    }
    if (numKeyHeads <= 0 || numValueHeads <= 0 || numValueHeads % numKeyHeads != 0) {
      throw new IllegalArgumentException("value heads must be a positive multiple of key heads");
    }
    this.numKeyHeads = numKeyHeads;
    this.numValueHeads = numValueHeads;
    this.keyHeadDim = keyHeadDim;
    this.valueHeadDim = valueHeadDim;
    this.keyDim = numKeyHeads * keyHeadDim;
    this.valueDim = numValueHeads * valueHeadDim;
    if (this.convDim != this.keyDim * 2 + this.valueDim) {
      throw new IllegalArgumentException(
        "conv channels " + this.convDim + " != 2*key + value " + (this.keyDim * 2 + this.valueDim));
    }
    this.headGroup = numValueHeads / numKeyHeads;
    this.queryScale = (float) (1.0 / Math.sqrt(keyHeadDim));
    this.normEps = normEps;
    this.layerIndex = layerIndex;
    this.aLog = requireVector(aLog, numValueHeads, "A_log");
    this.dtBias = requireVector(dtBias, numValueHeads, "dt_bias");
    this.normWeight = requireNonNull(normWeight, "normWeight");
  }

  private static void l2Normalize(
    final float[] data,
    final int tokens,
    final int heads,
    final int dim,
    final float scale
  ) {
    for (int t = 0; t < tokens; t++) {
      for (int h = 0; h < heads; h++) {
        int base = (t * heads + h) * dim;
        float sum = 0f;
        for (int i = 0; i < dim; i++) {
          float v = data[base + i];
          sum += v * v;
        }
        float inv = (float) (scale / Math.sqrt(sum + 1e-6));
        for (int i = 0; i < dim; i++) {
          data[base + i] *= inv;
        }
      }
    }
  }

  private static float[] kernelOf(final Tensor convWeight) {
    return convWeight.toFloatArray();
  }

  private static float[] requireVector(final Tensor tensor, final int length, final String name) {
    requireNonNull(tensor, name);
    if (tensor.numel() != length) {
      throw new IllegalArgumentException(name + " length " + tensor.numel() + " != " + length);
    }
    return tensor.toFloatArray();
  }

  private static float[] sigmoid(final float[] values) {
    float[] out = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = (float) (1.0 / (1.0 + Math.exp(-values[i])));
    }
    return out;
  }

  private static float silu(final float value) {
    return (float) (value / (1.0 + Math.exp(-value)));
  }

  private static float softplus(final float value) {
    if (value > 20f) {
      return value;
    }
    return (float) Math.log1p(Math.exp(value));
  }

  /**
   * One linear-attention pass. {@code hiddenStates} is {@code [tokens, hidden]}.
   *
   * @param hiddenStates mixer input
   * @param ctx          step context (optional delta arena and sequence ids)
   * @return {@code [tokens, hidden]}
   */
  public Tensor forward(final Tensor hiddenStates, final Context ctx) {
    requireNonNull(ctx, "ctx");
    int tokens = hiddenStates.size(0);
    float[] mixed = this.qkvProj.forward(hiddenStates, ctx).toFloatArray();
    this.causalConv(mixed, tokens, ctx);
    float[] query = new float[tokens * this.keyDim];
    float[] key = new float[tokens * this.keyDim];
    float[] value = new float[tokens * this.valueDim];
    this.splitQkv(mixed, tokens, query, key, value);
    this.l2Normalize(query, tokens, this.numKeyHeads, this.keyHeadDim, this.queryScale);
    this.l2Normalize(key, tokens, this.numKeyHeads, this.keyHeadDim, 1f);

    float[] beta = sigmoid(this.betaProj.forward(hiddenStates, ctx).toFloatArray());
    float[] decay = this.decayOf(this.alphaProj.forward(hiddenStates, ctx).toFloatArray(), tokens);
    float[] core = new float[tokens * this.valueDim];
    this.recurrent(query, key, value, beta, decay, core, tokens, ctx);

    Tensor gated = this.gateNorm(
      Tensor.of(core, tokens, this.numValueHeads, this.valueHeadDim),
      this.zProj.forward(hiddenStates, ctx)
        .reshape(tokens, this.numValueHeads, this.valueHeadDim));
    return this.outProj.forward(
      gated.reshape(tokens, this.valueDim), ctx);
  }

  /**
   * Causal depthwise conv. Weight tap {@code 0} multiplies the oldest sample in the window.
   * State, when present, is the last {@code kernel - 1} raw inputs.
   */
  void causalConv(final float[] mixed, final int tokens, final Context ctx) {
    DeltaNetStateArena arena = ctx.deltaCache();
    int[] seqIds = ctx.seqIds();
    int stateLen = this.kernelSize - 1;
    if (arena == null || seqIds == null) {
      float[] state = new float[this.convDim * stateLen];
      this.convRange(mixed, 0, tokens, state, stateLen);
      return;
    }
    if (ctx.isPrefill()) {
      int[] cuQ = ctx.cuSeqlensQ();
      for (int b = 0; b < cuQ.length - 1; b++) {
        int start = cuQ[b];
        int end = cuQ[b + 1];
        if (end <= start) {
          continue;
        }
        this.convRange(
          mixed, start, end, arena.convState(seqIds[start], this.layerIndex), stateLen);
      }
      return;
    }
    for (int t = 0; t < tokens; t++) {
      this.convRange(mixed, t, t + 1, arena.convState(seqIds[t], this.layerIndex), stateLen);
    }
  }

  private void convRange(
    final float[] mixed,
    final int start,
    final int end,
    final float[] state,
    final int stateLen
  ) {
    int seq = end - start;
    float[] raw = new float[seq * this.convDim];
    System.arraycopy(mixed, start * this.convDim, raw, 0, raw.length);
    for (int t = 0; t < seq; t++) {
      int outRow = (start + t) * this.convDim;
      for (int c = 0; c < this.convDim; c++) {
        float acc = 0f;
        for (int k = 0; k < this.kernelSize; k++) {
          acc += this.convKernel[c * this.kernelSize + k]
            * this.sample(raw, state, t, c, k, stateLen);
        }
        mixed[outRow + c] = silu(acc);
      }
    }
    this.storeConvState(raw, seq, state, stateLen);
  }

  private float sample(
    final float[] raw,
    final float[] state,
    final int t,
    final int channel,
    final int tap,
    final int stateLen
  ) {
    int index = t + tap;
    if (index < stateLen) {
      return state[channel * stateLen + index];
    }
    return raw[(index - stateLen) * this.convDim + channel];
  }

  private void storeConvState(
    final float[] raw,
    final int seq,
    final float[] state,
    final int stateLen
  ) {
    float[] next = new float[this.convDim * stateLen];
    for (int c = 0; c < this.convDim; c++) {
      for (int s = 0; s < stateLen; s++) {
        int time = seq - stateLen + s;
        next[c * stateLen + s] = time >= 0
          ? raw[time * this.convDim + c]
          : state[c * stateLen + (stateLen + time)];
      }
    }
    System.arraycopy(next, 0, state, 0, next.length);
  }

  private void splitQkv(
    final float[] mixed,
    final int tokens,
    final float[] query,
    final float[] key,
    final float[] value
  ) {
    for (int t = 0; t < tokens; t++) {
      int row = t * this.convDim;
      System.arraycopy(mixed, row, query, t * this.keyDim, this.keyDim);
      System.arraycopy(mixed, row + this.keyDim, key, t * this.keyDim, this.keyDim);
      System.arraycopy(mixed, row + 2 * this.keyDim, value, t * this.valueDim, this.valueDim);
    }
  }

  private void recurrent(
    final float[] query,
    final float[] key,
    final float[] value,
    final float[] beta,
    final float[] decay,
    final float[] core,
    final int tokens,
    final Context ctx
  ) {
    DeltaNetStateArena arena = ctx.deltaCache();
    int[] seqIds = ctx.seqIds();
    if (arena == null || seqIds == null) {
      float[] state = new float[this.numValueHeads * this.keyHeadDim * this.valueHeadDim];
      this.ruleRange(query, key, value, beta, decay, core, 0, tokens, state);
      return;
    }
    if (ctx.isPrefill()) {
      int[] cuQ = ctx.cuSeqlensQ();
      for (int b = 0; b < cuQ.length - 1; b++) {
        int start = cuQ[b];
        int end = cuQ[b + 1];
        if (end <= start) {
          continue;
        }
        this.ruleRange(
          query, key, value, beta, decay, core, start, end,
          arena.recurrentState(seqIds[start], this.layerIndex));
      }
      return;
    }
    for (int t = 0; t < tokens; t++) {
      this.ruleRange(
        query, key, value, beta, decay, core, t, t + 1,
        arena.recurrentState(seqIds[t], this.layerIndex));
    }
  }

  /**
   * Gated delta rule over tokens {@code [start, end)}. {@code state} is updated in place.
   */
  void ruleRange(
    final float[] query,
    final float[] key,
    final float[] value,
    final float[] beta,
    final float[] decay,
    final float[] core,
    final int start,
    final int end,
    final float[] state
  ) {
    int kDim = this.keyHeadDim;
    int vDim = this.valueHeadDim;
    for (int t = start; t < end; t++) {
      for (int h = 0; h < this.numValueHeads; h++) {
        int keyHead = h / this.headGroup;
        int qBase = (t * this.numKeyHeads + keyHead) * kDim;
        int vBase = (t * this.numValueHeads + h) * vDim;
        int sBase = (h * kDim) * vDim;
        float scale = (float) Math.exp(decay[t * this.numValueHeads + h]);
        for (int i = 0; i < kDim * vDim; i++) {
          state[sBase + i] *= scale;
        }
        float betaT = beta[t * this.numValueHeads + h];
        for (int v = 0; v < vDim; v++) {
          float predicted = 0f;
          for (int k = 0; k < kDim; k++) {
            predicted += state[sBase + k * vDim + v] * key[qBase + k];
          }
          float delta = (value[vBase + v] - predicted) * betaT;
          for (int k = 0; k < kDim; k++) {
            state[sBase + k * vDim + v] += key[qBase + k] * delta;
          }
          float out = 0f;
          for (int k = 0; k < kDim; k++) {
            out += state[sBase + k * vDim + v] * query[qBase + k];
          }
          core[vBase + v] = out;
        }
      }
    }
  }

  private Tensor gateNorm(final Tensor core, final Tensor gate) {
    Tensor normed = Ops.rmsNorm(core, this.normWeight, this.normEps, false);
    float[] data = normed.data();
    float[] z = gate.data();
    int zOff = gate.offset();
    for (int i = 0; i < data.length; i++) {
      data[i] *= silu(z[zOff + i]);
    }
    return normed;
  }

  private float[] decayOf(final float[] alpha, final int tokens) {
    float[] decay = new float[tokens * this.numValueHeads];
    for (int t = 0; t < tokens; t++) {
      for (int h = 0; h < this.numValueHeads; h++) {
        float a = (float) Math.exp(this.aLog[h]);
        decay[t * this.numValueHeads + h] =
          -a * softplus(alpha[t * this.numValueHeads + h] + this.dtBias[h]);
      }
    }
    return decay;
  }
}
