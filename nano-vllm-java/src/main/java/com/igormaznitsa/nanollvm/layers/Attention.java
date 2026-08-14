package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.engine.KvCacheArena;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;

public final class Attention {

  private static final int KEY_TILE = 64;
  private final int numHeads;
  private final int headDim;
  private final float scale;
  private final int numKvHeads;
  private final int repeats;
  private final int slidingWindow;
  private final int kvLayerIndex;
  private final boolean writeKv;

  public Attention(final int numHeads, final int headDim, final float scale, final int numKvHeads,
                   final int layerIndex) {
    this(numHeads, headDim, scale, numKvHeads, 0, layerIndex);
  }

  public Attention(
    final int numHeads,
    final int headDim,
    final float scale,
    final int numKvHeads,
    final int slidingWindow,
    final int layerIndex
  ) {
    this(numHeads, headDim, scale, numKvHeads, slidingWindow, layerIndex, true);
  }

  public Attention(
    final int numHeads,
    final int headDim,
    final float scale,
    final int numKvHeads,
    final int slidingWindow,
    final int kvLayerIndex,
    final boolean writeKv
  ) {
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.scale = scale;
    this.numKvHeads = numKvHeads;
    this.repeats = numHeads / numKvHeads;
    this.slidingWindow = Math.max(0, slidingWindow);
    this.kvLayerIndex = kvLayerIndex;
    this.writeKv = writeKv;
  }

  public Tensor forward(final Tensor q, final Tensor k, final Tensor v, final Context ctx) {
    KvCacheArena arena = requireNonNull(ctx.kvCache(), "KV cache arena not bound in Context");
    Tensor kCache = arena.k(this.kvLayerIndex);
    Tensor vCache = arena.v(this.kvLayerIndex);
    int[] slots = ctx.slotMapping();
    if (this.writeKv && kCache.numel() > 1 && vCache.numel() > 1 && slots != null
      && slots.length == k.size(0)) {
      this.storeKvCache(k, v, slots, kCache, vCache);
    }
    if (!this.writeKv) {
      return this.attendFromCache(q, ctx, kCache, vCache, slots);
    }
    if (ctx.isPrefill()) {
      if (ctx.blockTables() != null) {
        return this.prefillWithCache(q, ctx, kCache, vCache);
      }
      return this.prefillDense(q, k, v, ctx);
    }
    return this.decode(q, ctx, kCache, vCache);
  }

  private Tensor attendFromCache(
    final Tensor q,
    final Context ctx,
    final Tensor kCache,
    final Tensor vCache,
    final int[] slots
  ) {
    if (!ctx.isPrefill()) {
      return this.decode(q, ctx, kCache, vCache);
    }
    if (ctx.blockTables() != null) {
      return this.prefillWithCache(q, ctx, kCache, vCache);
    }
    if (slots == null || slots.length != q.size(0)) {
      throw new IllegalStateException(
        "shared-KV attention needs slot mapping or block tables to read the producer cache");
    }
    Tensor[] gathered = this.gatherKvCache(slots, kCache, vCache);
    return this.prefillDense(q, gathered[0], gathered[1], ctx);
  }

  private Tensor[] gatherKvCache(final int[] slotMapping, final Tensor kCache,
                                 final Tensor vCache) {
    int n = slotMapping.length;
    int d = this.numKvHeads * this.headDim;
    Tensor k = Tensor.zeros(n, this.numKvHeads, this.headDim);
    Tensor v = Tensor.zeros(n, this.numKvHeads, this.headDim);
    for (int i = 0; i < n; i++) {
      int slot = slotMapping[i];
      if (slot < 0) {
        continue;
      }
      System.arraycopy(kCache.data(), kCache.offset() + slot * d, k.data(), i * d, d);
      System.arraycopy(vCache.data(), vCache.offset() + slot * d, v.data(), i * d, d);
    }
    return new Tensor[] {k, v};
  }

  private void storeKvCache(
    final Tensor key,
    final Tensor value,
    final int[] slotMapping,
    final Tensor kCache,
    final Tensor vCache
  ) {
    int n = key.size(0);
    int d = this.numKvHeads * this.headDim;
    for (int i = 0; i < n; i++) {
      int slot = slotMapping[i];
      if (slot < 0) {
        continue;
      }
      System.arraycopy(key.data(), key.offset() + i * d, kCache.data(),
        kCache.offset() + slot * d, d);
      System.arraycopy(value.data(), value.offset() + i * d, vCache.data(),
        vCache.offset() + slot * d, d);
    }
  }

  private Tensor prefillDense(final Tensor q, final Tensor k, final Tensor v, final Context ctx) {
    int[] cuQ = ctx.cuSeqlensQ();
    int[] cuK = ctx.cuSeqlensK();
    int batch = cuQ.length - 1;
    Tensor out = Tensor.zeros(q.size(0), this.numHeads, this.headDim);
    for (int b = 0; b < batch; b++) {
      int qStart = cuQ[b];
      int qEnd = cuQ[b + 1];
      int kStart = cuK[b];
      int kEnd = cuK[b + 1];
      this.attendRange(q, k, v, out, qStart, qEnd, kStart, kEnd - kStart, true);
    }
    return out;
  }

  private Tensor prefillWithCache(final Tensor q, final Context ctx, final Tensor kCache,
                                  final Tensor vCache) {
    int[] cuQ = ctx.cuSeqlensQ();
    int[] cuK = ctx.cuSeqlensK();
    int batch = cuQ.length - 1;
    int[][] blockTables = ctx.blockTables();
    Tensor out = Tensor.zeros(q.size(0), this.numHeads, this.headDim);
    int d = this.numKvHeads * this.headDim;
    int blockSize = kCache.size(1);

    for (int b = 0; b < batch; b++) {
      int qStart = cuQ[b];
      int qEnd = cuQ[b + 1];
      int kLen = cuK[b + 1] - cuK[b];
      Tensor k = Tensor.zeros(kLen, this.numKvHeads, this.headDim);
      Tensor v = Tensor.zeros(kLen, this.numKvHeads, this.headDim);
      int[] table = blockTables[b];
      for (int t = 0; t < kLen; t++) {
        int blockId = table[t / blockSize];
        int offset = t % blockSize;
        int slot = blockId * blockSize + offset;
        System.arraycopy(kCache.data(), kCache.offset() + slot * d, k.data(), t * d, d);
        System.arraycopy(vCache.data(), vCache.offset() + slot * d, v.data(), t * d, d);
      }
      this.attendRange(q, k, v, out, qStart, qEnd, 0, kLen, true);
    }
    return out;
  }

  private Tensor decode(final Tensor q, final Context ctx, final Tensor kCache,
                        final Tensor vCache) {
    int bs = q.size(0);
    int[] contextLens = ctx.contextLens();
    int[][] blockTables = ctx.blockTables();
    int d = this.numKvHeads * this.headDim;
    int blockSize = kCache.size(1);
    Tensor out = Tensor.zeros(bs, this.numHeads, this.headDim);

    for (int b = 0; b < bs; b++) {
      int kLen = contextLens[b];
      Tensor k = Tensor.zeros(kLen, this.numKvHeads, this.headDim);
      Tensor v = Tensor.zeros(kLen, this.numKvHeads, this.headDim);
      int[] table = blockTables[b];
      for (int t = 0; t < kLen; t++) {
        int blockId = table[t / blockSize];
        int offset = t % blockSize;
        int slot = blockId * blockSize + offset;
        System.arraycopy(kCache.data(), kCache.offset() + slot * d, k.data(), t * d, d);
        System.arraycopy(vCache.data(), vCache.offset() + slot * d, v.data(), t * d, d);
      }
      this.attendRange(q, k, v, out, b, b + 1, 0, kLen, false);
    }
    return out;
  }

  private void attendRange(
    final Tensor q, final Tensor k, final Tensor v, final Tensor out,
    final int qStart, final int qEnd, final int kIndexBase, final int kLen, final boolean causal
  ) {
    int qLen = qEnd - qStart;
    int work = qLen * this.numHeads;
    for (int job = 0; job < work; job++) {
      int qi = job / this.numHeads;
      int h = job % this.numHeads;
      int qPos = qStart + qi;
      int causalEnd = causal ? (kLen - qLen + qi + 1) : kLen;
      int causalStart = 0;
      if (this.slidingWindow > 0) {
        int absQ = causal ? (kLen - qLen + qi) : (kLen - 1);
        causalStart = Math.max(0, absQ - this.slidingWindow + 1);
      }
      int kvh = h / this.repeats;
      int qBase = q.offset() + (qPos * this.numHeads + h) * this.headDim;
      int oBase = (qPos * this.numHeads + h) * this.headDim;

      float m = Float.NEGATIVE_INFINITY;
      float l = 0f;
      float[] acc = new float[this.headDim];

      for (int k0 = causalStart; k0 < causalEnd; k0 += KEY_TILE) {
        int k1 = Math.min(causalEnd, k0 + KEY_TILE);
        float tileMax = Float.NEGATIVE_INFINITY;
        float[] tileScores = new float[k1 - k0];
        for (int kj = k0; kj < k1; kj++) {
          int kBase = k.offset() + ((kIndexBase + kj) * this.numKvHeads + kvh) * this.headDim;
          float score = VectorMath.dot(q.data(), qBase, k.data(), kBase, this.headDim) * this.scale;
          tileScores[kj - k0] = score;
          if (score > tileMax) {
            tileMax = score;
          }
        }
        float mNew = Math.max(m, tileMax);
        float alpha = m == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(m - mNew);
        float lNew = l * alpha;
        for (int d = 0; d < this.headDim; d++) {
          acc[d] *= alpha;
        }
        for (int kj = k0; kj < k1; kj++) {
          float w = (float) Math.exp(tileScores[kj - k0] - mNew);
          lNew += w;
          int vBase = v.offset() + ((kIndexBase + kj) * this.numKvHeads + kvh) * this.headDim;
          for (int d = 0; d < this.headDim; d++) {
            acc[d] += w * v.data()[vBase + d];
          }
        }
        m = mNew;
        l = lNew;
      }

      float inv = l == 0f ? 0f : 1f / l;
      for (int d = 0; d < this.headDim; d++) {
        out.data()[oBase + d] = acc[d] * inv;
      }
    }
  }
}
