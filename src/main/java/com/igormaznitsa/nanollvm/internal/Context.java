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

  public void bindKvCache(final KvCacheArena arena) {
    this.kvCache = arena;
  }

  public void bindConvCache(final ConvStateArena arena) {
    this.convCache = arena;
  }

  public void bindMatmul(final MatmulRuntime runtime) {
    this.matmul = runtime;
  }

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

  public boolean isPrefill() {
    return this.prefill;
  }

  public int[] cuSeqlensQ() {
    return this.cuSeqlensQ;
  }

  public int[] cuSeqlensK() {
    return this.cuSeqlensK;
  }

  public int maxSeqlenQ() {
    return this.maxSeqlenQ;
  }

  public int maxSeqlenK() {
    return this.maxSeqlenK;
  }

  public int[] slotMapping() {
    return this.slotMapping;
  }

  public int[] contextLens() {
    return this.contextLens;
  }

  public int[][] blockTables() {
    return this.blockTables;
  }

  public int[] seqIds() {
    return this.seqIds;
  }

  public KvCacheArena kvCache() {
    return this.kvCache;
  }

  public ConvStateArena convCache() {
    return this.convCache;
  }

  public MatmulRuntime matmul() {
    return this.matmul;
  }
}
