package com.igormaznitsa.nanollvm.internal;

import com.igormaznitsa.nanollvm.engine.ConvStateArena;
import com.igormaznitsa.nanollvm.engine.KvCacheArena;

public final class Context {

  private static final ThreadLocal<Context> CURRENT = ThreadLocal.withInitial(Context::new);

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

  public static Context get() {
    return CURRENT.get();
  }

  public static void bindKvCache(final KvCacheArena arena) {
    CURRENT.get().kvCache = arena;
  }

  public static void bindConvCache(final ConvStateArena arena) {
    CURRENT.get().convCache = arena;
  }

  public static void set(
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
    Context ctx = CURRENT.get();
    ctx.prefill = isPrefill;
    ctx.cuSeqlensQ = cuSeqlensQ;
    ctx.cuSeqlensK = cuSeqlensK;
    ctx.maxSeqlenQ = maxSeqlenQ;
    ctx.maxSeqlenK = maxSeqlenK;
    ctx.slotMapping = slotMapping;
    ctx.contextLens = contextLens;
    ctx.blockTables = blockTables;
    ctx.seqIds = seqIds;
  }

  public static void reset() {
    CURRENT.set(new Context());
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
}
