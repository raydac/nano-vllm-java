package com.igormaznitsa.nanollvm.internal;

import com.igormaznitsa.nanollvm.engine.ConvStateArena;
import com.igormaznitsa.nanollvm.engine.KvCacheArena;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;

/**
 * Step-scoped engine state for one transformer forward: KV / conv arenas, matmul runtime,
 * and varlen attention metadata. Owned and passed explicitly by {@code Transformer} — not
 * thread-local.
 */
public final class Context {

  private boolean prefill;
  private int[] cuSeqlensQ;
  private int[] cuSeqlensK;
  private int maxSeqlenQ;
  private int maxSeqlenK;
  private int[] slotMapping;
  private int[] contextLens;
  private int[][] blockTables;
  private int[] seqIds;
  private KvCacheArena kvCache;
  private ConvStateArena convCache;
  private MatmulRuntime matmul;

  /**
   * Binds the paged KV arena for this step (chat). {@code null} clears the binding.
   */
  public void bindKvCache(final KvCacheArena arena) {
    this.kvCache = arena;
  }

  /**
   * Binds the LFM2 short-conv state arena for this step. {@code null} clears the binding.
   */
  public void bindConvCache(final ConvStateArena arena) {
    this.convCache = arena;
  }

  /**
   * Binds the engine matmul runtime (parallel GEMM / attention jobs). {@code null} means
   * sequential kernels.
   */
  public void bindMatmul(final MatmulRuntime runtime) {
    this.matmul = runtime;
  }

  /**
   * Installs varlen attention metadata for this prefill or decode step.
   *
   * @param isPrefill   {@code true} when packing new prompt tokens
   * @param cuSeqlensQ  cumulative query lengths
   * @param cuSeqlensK  cumulative key lengths
   * @param maxSeqlenQ  max query length in the batch
   * @param maxSeqlenK  max key length in the batch
   * @param slotMapping KV slot ids for writes
   * @param contextLens per-sequence context lengths
   * @param blockTables per-sequence page tables
   * @param seqIds      engine sequence ids
   */
  public void set(
    final boolean isPrefill,
    final int[] cuSeqlensQ,
    final int[] cuSeqlensK,
    final int maxSeqlenQ,
    final int maxSeqlenK,
    final int[] slotMapping,
    final int[] contextLens,
    final int[][] blockTables,
    final int[] seqIds
  ) {
    this.prefill = isPrefill;
    this.cuSeqlensQ = cuSeqlensQ;
    this.cuSeqlensK = cuSeqlensK;
    this.maxSeqlenQ = maxSeqlenQ;
    this.maxSeqlenK = maxSeqlenK;
    this.slotMapping = slotMapping;
    this.contextLens = contextLens;
    this.blockTables = blockTables;
    this.seqIds = seqIds;
  }

  /**
   * Drops varlen metadata and arena bindings after the step.
   */
  public void clear() {
    this.prefill = false;
    this.cuSeqlensQ = null;
    this.cuSeqlensK = null;
    this.maxSeqlenQ = 0;
    this.maxSeqlenK = 0;
    this.slotMapping = null;
    this.contextLens = null;
    this.blockTables = null;
    this.seqIds = null;
    this.kvCache = null;
    this.convCache = null;
    this.matmul = null;
  }

  /**
   * {@code true} when this step is packing prompt tokens (not decode).
   */
  public boolean isPrefill() {
    return this.prefill;
  }

  /** Cumulative query lengths for the current batch, or {@code null} when unset. */
  public int[] cuSeqlensQ() {
    return this.cuSeqlensQ;
  }

  /** Cumulative key lengths for the current batch, or {@code null} when unset. */
  public int[] cuSeqlensK() {
    return this.cuSeqlensK;
  }

  /** Max query length in the current batch. */
  public int maxSeqlenQ() {
    return this.maxSeqlenQ;
  }

  /** Max key length in the current batch. */
  public int maxSeqlenK() {
    return this.maxSeqlenK;
  }

  /** KV write-slot mapping, or {@code null} when unset. */
  public int[] slotMapping() {
    return this.slotMapping;
  }

  /** Per-sequence context lengths, or {@code null} when unset. */
  public int[] contextLens() {
    return this.contextLens;
  }

  /** Per-sequence page tables, or {@code null} when unset. */
  public int[][] blockTables() {
    return this.blockTables;
  }

  /** Engine sequence ids for the current batch, or {@code null} when unset. */
  public int[] seqIds() {
    return this.seqIds;
  }

  /** Paged KV arena bound for this step, or {@code null}. */
  public KvCacheArena kvCache() {
    return this.kvCache;
  }

  /** Short-conv state arena bound for this step, or {@code null}. */
  public ConvStateArena convCache() {
    return this.convCache;
  }

  /** Matmul runtime bound for this step, or {@code null} for sequential kernels. */
  public MatmulRuntime matmul() {
    return this.matmul;
  }
}
