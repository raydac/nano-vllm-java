package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.engine.KvCacheArena;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Arrays;

/**
 * Causal grouped-query attention with a paged KV cache. Used by chat architectures
 * (Qwen3, Gemma 3/4, Llama, LFM2). BERT uses {@link BidirectionalAttention} instead.
 *
 * <h2>Layout</h2>
 * Query is {@code [tokens, numHeads, headDim]}; key/value are {@code [tokens, numKvHeads, headDim]}.
 * When {@code numHeads > numKvHeads}, each KV head is reused {@code numHeads / numKvHeads} times
 * (GQA). Scores are {@code q·k * scale} with online softmax over 64-key tiles.
 *
 * <h2>Prefill vs decode</h2>
 * Driven by {@link Context#isPrefill()} and the bound {@link KvCacheArena}:
 * <ul>
 *   <li><b>Prefill, no block tables</b> — attend over the dense {@code k}/{@code v} of this batch
 *       (varlen ranges from {@code cuSeqlensQ}/{@code cuSeqlensK}).</li>
 *   <li><b>Prefill, with block tables</b> — write new KV into arena slots, then attend in place
 *       over paged cache (no dense page copy).</li>
 *   <li><b>Decode</b> — one query token per sequence; attend over the sequence's cached keys
 *       via {@link Context#blockTables()} and {@link Context#contextLens()}.</li>
 * </ul>
 *
 * <h2>Shared KV (Gemma 4)</h2>
 * A layer with {@code writeKv == false} does not project or store K/V. It reads the producer
 * layer's cache at {@code kvLayerIndex}. A shared layer still needs slot mapping or block tables
 * so it can find those pages.
 *
 * <h2>Sliding window</h2>
 * {@code slidingWindow > 0} restricts the attended key range to the last {@code slidingWindow}
 * positions (Gemma local layers). {@code 0} is full causal context.
 *
 * <p><strong>Thread safety:</strong> {@link #forward} is exclusive (one generate thread).
 * Bound {@link MatmulRuntime} may run disjoint head×query jobs in parallel (head-major so
 * each chunk covers a full query range and causal work stays balanced); jobs are leaf
 * kernels and do not re-enter the pool.
 *
 * @see BidirectionalAttention
 * @see com.igormaznitsa.nanollvm.engine.KvCacheArena
 */
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

  /**
   * Causal GQA that writes KV into {@code layerIndex} of the arena, with no sliding window.
   *
   * @param numHeads   query heads
   * @param headDim    per-head width
   * @param scale      score multiplier ({@code 1/√d} or architecture-specific)
   * @param numKvHeads key/value heads ({@code <= numHeads}; GQA when smaller)
   * @param layerIndex KV arena layer this projection writes
   */
  public Attention(final int numHeads, final int headDim, final float scale, final int numKvHeads,
                   final int layerIndex) {
    this(numHeads, headDim, scale, numKvHeads, 0, layerIndex);
  }

  /**
   * Causal GQA that writes KV into {@code layerIndex}, optionally windowed.
   *
   * @param numHeads      query heads
   * @param headDim       per-head width
   * @param scale         score multiplier
   * @param numKvHeads    key/value heads
   * @param slidingWindow max attended keys; {@code 0} = full causal context
   * @param layerIndex    KV arena layer this projection writes
   */
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

  /**
   * Causal GQA with an explicit KV producer layer and write flag (Gemma 4 shared-KV layers).
   *
   * @param numHeads      query heads
   * @param headDim       per-head width
   * @param scale         score multiplier
   * @param numKvHeads    key/value heads
   * @param slidingWindow max attended keys; {@code 0} = full causal context
   * @param kvLayerIndex  arena layer to read (and write when {@code writeKv})
   * @param writeKv       {@code false} to attend from a producer cache without storing
   * @since 1.1.0
   */
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

  /**
   * Flattens a paged block table of length {@code kLen} into arena slot indices
   * ({@code blockId * blockSize + offset}).
   */
  private static int[] pagedSlots(final int[] table, final int kLen, final int blockSize) {
    int[] slots = new int[kLen];
    for (int t = 0; t < kLen; t++) {
      slots[t] = table[t / blockSize] * blockSize + (t % blockSize);
    }
    return slots;
  }

  /**
   * One attention pass over a scheduled batch. Requires a {@link KvCacheArena} bound on
   * {@code ctx}. When this layer writes KV and slot mapping matches the token count, new
   * {@code k}/{@code v} are stored before attending.
   *
   * @param q   queries {@code [tokens, numHeads, headDim]}
   * @param k   keys {@code [tokens, numKvHeads, headDim]} (ignored when not writing KV)
   * @param v   values, same layout as {@code k}
   * @param ctx step context (arena, cu-seqlens, slots / block tables)
   * @return attended values {@code [tokens, numHeads, headDim]}
   * @throws NullPointerException  if {@code ctx} has no KV arena
   * @throws IllegalStateException if a shared-KV layer has neither slots nor block tables
   */
  public Tensor forward(final Tensor q, final Tensor k, final Tensor v, final Context ctx) {
    KvCacheArena arena = requireNonNull(ctx.kvCache(), "KV cache arena not bound in Context");
    Tensor kCache = arena.getKeyCache(this.kvLayerIndex);
    Tensor vCache = arena.getValueCache(this.kvLayerIndex);
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

  /**
   * Shared-KV path: attend using the producer layer's cache, never the caller's {@code k}/{@code v}.
   */
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

  /**
   * Copies KV rows from arena slots into dense {@code [n, numKvHeads, headDim]} tensors.
   * Negative slots are left as zeros (padding).
   */
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

  /**
   * Writes each token's K/V into the arena at {@code slotMapping[i]}. Negative slots are skipped.
   */
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

  /**
   * Prefill over dense K/V of this batch, one varlen sequence at a time.
   */
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
      this.attendRange(q, k, v, out, qStart, qEnd, kStart, kEnd - kStart, true, null, ctx);
    }
    return out;
  }

  /**
   * Prefill that attends in place over paged cache slots (no dense K/V copy).
   */
  private Tensor prefillWithCache(final Tensor q, final Context ctx, final Tensor kCache,
                                  final Tensor vCache) {
    int[] cuQ = ctx.cuSeqlensQ();
    int[] cuK = ctx.cuSeqlensK();
    int batch = cuQ.length - 1;
    int[][] blockTables = ctx.blockTables();
    Tensor out = Tensor.zeros(q.size(0), this.numHeads, this.headDim);
    int blockSize = kCache.size(1);

    for (int b = 0; b < batch; b++) {
      int qStart = cuQ[b];
      int qEnd = cuQ[b + 1];
      int kLen = cuK[b + 1] - cuK[b];
      int[] slots = pagedSlots(blockTables[b], kLen, blockSize);
      this.attendRange(q, kCache, vCache, out, qStart, qEnd, 0, kLen, true, slots, ctx);
    }
    return out;
  }

  /**
   * Decode: one query token per sequence, attend over that sequence's cached keys.
   */
  private Tensor decode(final Tensor q, final Context ctx, final Tensor kCache,
                        final Tensor vCache) {
    int bs = q.size(0);
    int[] contextLens = ctx.contextLens();
    int[][] blockTables = ctx.blockTables();
    int blockSize = kCache.size(1);
    Tensor out = Tensor.zeros(bs, this.numHeads, this.headDim);

    for (int b = 0; b < bs; b++) {
      int kLen = contextLens[b];
      int[] slots = pagedSlots(blockTables[b], kLen, blockSize);
      this.attendRange(q, kCache, vCache, out, b, b + 1, 0, kLen, false, slots, ctx);
    }
    return out;
  }

  /**
   * Online-softmax attention for query tokens {@code [qStart, qEnd)} over {@code kLen} keys.
   * When {@code causal} is true, query {@code i} sees keys up to its own position. {@code kvSlots}
   * {@code null} means dense keys starting at {@code kIndexBase}; otherwise keys are read from
   * those arena slots. Head×query jobs are independent and run on {@code ctx.matmul()} when the
   * work is large enough.
   */
  private void attendRange(
    final Tensor q, final Tensor k, final Tensor v, final Tensor out,
    final int qStart, final int qEnd, final int kIndexBase, final int kLen, final boolean causal,
    final int[] kvSlots,
    final Context ctx
  ) {
    int qLen = qEnd - qStart;
    int work = qLen * this.numHeads;
    MatmulRuntime runtime = ctx.matmul() == null ? MatmulRuntime.sequential() : ctx.matmul();
    int cost = work * Math.max(kLen, 1);
    if (cost < 256) {
      this.attendJobs(0, work, q, k, v, out, qStart, qLen, kIndexBase, kLen, causal, kvSlots);
      return;
    }
    runtime.parallelRanges(
      work,
      (start, end) -> this.attendJobs(
        start, end, q, k, v, out, qStart, qLen, kIndexBase, kLen, causal, kvSlots));
  }

  private void attendJobs(
    final int jobStart,
    final int jobEnd,
    final Tensor q,
    final Tensor k,
    final Tensor v,
    final Tensor out,
    final int qStart,
    final int qLen,
    final int kIndexBase,
    final int kLen,
    final boolean causal,
    final int[] kvSlots
  ) {
    float[] acc = new float[this.headDim];
    float[] tileScores = new float[KEY_TILE];
    float[] qData = q.data();
    float[] kData = k.data();
    float[] vData = v.data();
    float[] oData = out.data();

    for (int job = jobStart; job < jobEnd; job++) {
      int h = job / qLen;
      int qi = job % qLen;
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

      Arrays.fill(acc, 0f);
      float m = Float.NEGATIVE_INFINITY;
      float l = 0f;

      for (int k0 = causalStart; k0 < causalEnd; k0 += KEY_TILE) {
        int k1 = Math.min(causalEnd, k0 + KEY_TILE);
        float tileMax = Float.NEGATIVE_INFINITY;
        for (int kj = k0; kj < k1; kj++) {
          float score = VectorMath.dot(
            qData, qBase, kData, this.kvHeadBase(k, kIndexBase, kj, kvh, kvSlots), this.headDim)
            * this.scale;
          tileScores[kj - k0] = score;
          if (score > tileMax) {
            tileMax = score;
          }
        }
        float mNew = Math.max(m, tileMax);
        float alpha = m == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(m - mNew);
        float lNew = l * alpha;
        VectorMath.scale(acc, 0, alpha, acc, 0, this.headDim);
        for (int kj = k0; kj < k1; kj++) {
          float w = (float) Math.exp(tileScores[kj - k0] - mNew);
          lNew += w;
          VectorMath.axpy(
            acc, 0, w, vData, this.kvHeadBase(v, kIndexBase, kj, kvh, kvSlots), this.headDim);
        }
        m = mNew;
        l = lNew;
      }

      float inv = l == 0f ? 0f : 1f / l;
      VectorMath.scale(acc, 0, inv, oData, oBase, this.headDim);
    }
  }

  /**
   * Byte offset of KV-head {@code kvh} for key index {@code kj} in {@code cache}.
   */
  private int kvHeadBase(
    final Tensor cache,
    final int kIndexBase,
    final int kj,
    final int kvh,
    final int[] kvSlots
  ) {
    int token = kvSlots != null ? kvSlots[kj] : kIndexBase + kj;
    return cache.offset() + (token * this.numKvHeads + kvh) * this.headDim;
  }
}
