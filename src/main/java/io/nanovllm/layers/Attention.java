package io.nanovllm.layers;

import io.nanovllm.tensor.Tensor;
import io.nanovllm.tensor.VectorMath;
import io.nanovllm.utils.Context;

public final class Attention {

  private static final int KEY_TILE = 64;
  private final int numHeads;
  private final int headDim;
  private final float scale;
  private final int numKvHeads;
  private final int repeats;
  private final int slidingWindow;
  private Tensor kCache = Tensor.zeros(1);
  private Tensor vCache = Tensor.zeros(1);

  public Attention(int numHeads, int headDim, float scale, int numKvHeads) {
    this(numHeads, headDim, scale, numKvHeads, 0);
  }

  public Attention(int numHeads, int headDim, float scale, int numKvHeads, int slidingWindow) {
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.scale = scale;
    this.numKvHeads = numKvHeads;
    this.repeats = numHeads / numKvHeads;
    this.slidingWindow = Math.max(0, slidingWindow);
  }

  public void setCaches(Tensor kCache, Tensor vCache) {
    this.kCache = kCache;
    this.vCache = vCache;
  }

  public Tensor kCache() {
    return this.kCache;
  }

  public Tensor vCache() {
    return this.vCache;
  }

  public Tensor forward(Tensor q, Tensor k, Tensor v) {
    Context ctx = Context.get();
    int[] slots = ctx.slotMapping();
    if (this.kCache.numel() > 1 && this.vCache.numel() > 1 && slots != null &&
        slots.length == k.size(0)) {
      this.storeKvCache(k, v, slots);
    }
    if (ctx.isPrefill()) {
      if (ctx.blockTables() != null) {
        return this.prefillWithCache(q, ctx);
      }
      return this.prefillDense(q, k, v, ctx);
    }
    return this.decode(q, ctx);
  }

  private void storeKvCache(Tensor key, Tensor value, int[] slotMapping) {
    int n = key.size(0);
    int d = this.numKvHeads * this.headDim;
    for (int i = 0; i < n; i++) {
      int slot = slotMapping[i];
      if (slot < 0) {
        continue;
      }
      System.arraycopy(key.data(), key.offset() + i * d, this.kCache.data(),
          this.kCache.offset() + slot * d, d);
      System.arraycopy(value.data(), value.offset() + i * d, this.vCache.data(),
          this.vCache.offset() + slot * d, d);
    }
  }

  private Tensor prefillDense(Tensor q, Tensor k, Tensor v, Context ctx) {
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

  private Tensor prefillWithCache(Tensor q, Context ctx) {
    int[] cuQ = ctx.cuSeqlensQ();
    int[] cuK = ctx.cuSeqlensK();
    int batch = cuQ.length - 1;
    int[][] blockTables = ctx.blockTables();
    Tensor out = Tensor.zeros(q.size(0), this.numHeads, this.headDim);
    int d = this.numKvHeads * this.headDim;
    int blockSize = this.kCache.size(1);

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
        System.arraycopy(this.kCache.data(), this.kCache.offset() + slot * d, k.data(), t * d, d);
        System.arraycopy(this.vCache.data(), this.vCache.offset() + slot * d, v.data(), t * d, d);
      }
      this.attendRange(q, k, v, out, qStart, qEnd, 0, kLen, true);
    }
    return out;
  }

  private Tensor decode(Tensor q, Context ctx) {
    int bs = q.size(0);
    int[] contextLens = ctx.contextLens();
    int[][] blockTables = ctx.blockTables();
    int d = this.numKvHeads * this.headDim;
    int blockSize = this.kCache.size(1);
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
        System.arraycopy(this.kCache.data(), this.kCache.offset() + slot * d, k.data(), t * d, d);
        System.arraycopy(this.vCache.data(), this.vCache.offset() + slot * d, v.data(), t * d, d);
      }
      this.attendRange(q, k, v, out, b, b + 1, 0, kLen, false);
    }
    return out;
  }

  private void attendRange(
      Tensor q, Tensor k, Tensor v, Tensor out,
      int qStart, int qEnd, int kIndexBase, int kLen, boolean causal
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
