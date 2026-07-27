package io.nanovllm.utils;

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

  public static Context get() {
    return CURRENT.get();
  }

  public static void set(
      boolean isPrefill,
      int[] cuSeqlensQ,
      int[] cuSeqlensK,
      int maxSeqlenQ,
      int maxSeqlenK,
      int[] slotMapping,
      int[] contextLens,
      int[][] blockTables
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
}
