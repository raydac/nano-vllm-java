package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled Tornado plan per weight matrix and token count. Activations are copied into a
 * plan-owned buffer so a new result array does not recompile the kernel or re-upload weights.
 *
 * @since 1.4.0
 */
final class TornadoGemvExecutor {

  private static final int MAX_CACHED_PLANS = 256;
  private static final String GRAPH_NAME = "nanollvm-gemv";
  private static final String TASK_NAME = "gemv";
  private static final String SCHEDULER_KEY = GRAPH_NAME + "." + TASK_NAME;
  private static final ReentrantLock EXEC_LOCK = new ReentrantLock();
  private static final Map<WeightKey, GemvPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<WeightKey, GemvPlan> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private TornadoGemvExecutor() {
  }

  static void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    gemvRows(x, xOff, w, wOff, bias, y, yOff, 1, in, out1 - out0, out0, out1);
  }

  static void gemvRows(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int rows, final int in, final int rowStride,
    final int out0, final int out1
  ) {
    int outCount = out1 - out0;
    if (rows <= 0 || outCount <= 0 || in < 0) {
      return;
    }
    if (outCount > Integer.MAX_VALUE / rows) {
      throw new IllegalArgumentException("TornadoVM GEMV launch exceeds Integer range");
    }
    WeightKey key = WeightKey.of(w, wOff, bias, in, out0, outCount, rows);
    EXEC_LOCK.lock();
    try {
      GemvPlan plan = CACHED_PLANS.get(key);
      if (plan == null) {
        plan = GemvPlan.compile(key);
        CACHED_PLANS.put(key, plan);
      }
      plan.execute(x, xOff, y, yOff, rowStride);
    } catch (TornadoExecutionPlanException e) {
      throw new IllegalStateException("TornadoVM GEMV failed", e);
    } finally {
      EXEC_LOCK.unlock();
    }
  }

  private static final class WeightKey {

    private final float[] weight;
    private final int weightOffset;
    private final float[] bias;
    private final int in;
    private final int out0;
    private final int outCount;
    private final int rows;

    private WeightKey(
      final float[] weight, final int weightOffset,
      final float[] bias,
      final int in, final int out0, final int outCount, final int rows
    ) {
      this.weight = weight;
      this.weightOffset = weightOffset;
      this.bias = bias;
      this.in = in;
      this.out0 = out0;
      this.outCount = outCount;
      this.rows = rows;
    }

    static WeightKey of(
      final float[] weight, final int weightOffset,
      final float[] bias,
      final int in, final int out0, final int outCount, final int rows
    ) {
      return new WeightKey(weight, weightOffset, bias, in, out0, outCount, rows);
    }

    float[] weight() {
      return this.weight;
    }

    int weightOffset() {
      return this.weightOffset;
    }

    int in() {
      return this.in;
    }

    int out0() {
      return this.out0;
    }

    int outCount() {
      return this.outCount;
    }

    int rows() {
      return this.rows;
    }

    int work() {
      return this.rows * this.outCount;
    }

    boolean hasBias() {
      return this.bias != null;
    }

    float[] biasArg() {
      return this.hasBias() ? this.bias : TornadoGemvKernels.NO_BIAS;
    }

    int hasBiasFlag() {
      return this.hasBias() ? 1 : 0;
    }

    @Override
    @SuppressWarnings("ReferenceEquality")
    public boolean equals(final Object other) {
      if (!(other instanceof WeightKey that)) {
        return false;
      }
      return this.weight == that.weight && this.weightOffset == that.weightOffset
        && this.bias == that.bias
        && this.in == that.in && this.out0 == that.out0
        && this.outCount == that.outCount && this.rows == that.rows;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
        System.identityHashCode(this.weight), this.weightOffset,
        System.identityHashCode(this.bias),
        this.in, this.out0, this.outCount, this.rows);
    }
  }

  private static final class GemvPlan implements AutoCloseable {

    private final TornadoExecutionPlan plan;
    private final float[] xScratch;
    private final float[] yScratch;
    private final int rows;
    private final int in;
    private final int out0;
    private final int outCount;

    private GemvPlan(
      final TornadoExecutionPlan plan,
      final float[] xScratch, final float[] yScratch,
      final int rows, final int in, final int out0, final int outCount
    ) {
      this.plan = plan;
      this.xScratch = xScratch;
      this.yScratch = yScratch;
      this.rows = rows;
      this.in = in;
      this.out0 = out0;
      this.outCount = outCount;
    }

    static GemvPlan compile(final WeightKey key) throws TornadoExecutionPlanException {
      float[] xScratch = new float[key.rows() * key.in()];
      float[] yScratch = new float[key.work()];
      try {
        return compileTiled(key, xScratch, yScratch);
      } catch (Throwable tiledFailed) {
        try {
          return compilePlain(key, xScratch, yScratch);
        } catch (Throwable plainFailed) {
          plainFailed.addSuppressed(tiledFailed);
          try {
            return compileLoopParallel(key, xScratch, yScratch);
          } catch (TornadoExecutionPlanException parallelFailed) {
            parallelFailed.addSuppressed(plainFailed);
            throw parallelFailed;
          }
        }
      }
    }

    private static GemvPlan compileTiled(
      final WeightKey key, final float[] xScratch, final float[] yScratch
    ) throws TornadoExecutionPlanException {
      int laneWidth = (int) localWorkSize(key.outCount());
      WorkerGrid1D worker = new WorkerGrid1D(key.work());
      worker.setLocalWork(laneWidth, 1, 1);
      GridScheduler scheduler = new GridScheduler(SCHEDULER_KEY, worker);
      TaskGraph taskGraph = taskGraph(key, xScratch, yScratch)
        .task(
          TASK_NAME,
          TornadoGemvKernels::gemvTiled,
          new KernelContext(),
          xScratch,
          key.weight(), key.weightOffset(),
          key.biasArg(), key.hasBiasFlag(),
          yScratch,
          key.in(), key.out0(), key.outCount(), laneWidth
        )
        .transferToHost(EVERY_EXECUTION, (Object) yScratch);
      return open(taskGraph, scheduler, xScratch, yScratch, key);
    }

    private static GemvPlan compilePlain(
      final WeightKey key, final float[] xScratch, final float[] yScratch
    ) throws TornadoExecutionPlanException {
      WorkerGrid1D worker = new WorkerGrid1D(key.work());
      worker.setLocalWork(localWorkSize(key.work()), 1, 1);
      GridScheduler scheduler = new GridScheduler(SCHEDULER_KEY, worker);
      TaskGraph taskGraph = taskGraph(key, xScratch, yScratch)
        .task(
          TASK_NAME,
          TornadoGemvKernels::gemvPlain,
          new KernelContext(),
          xScratch,
          key.weight(), key.weightOffset(),
          key.biasArg(), key.hasBiasFlag(),
          yScratch,
          key.in(), key.out0(), key.outCount()
        )
        .transferToHost(EVERY_EXECUTION, (Object) yScratch);
      return open(taskGraph, scheduler, xScratch, yScratch, key);
    }

    private static GemvPlan compileLoopParallel(
      final WeightKey key, final float[] xScratch, final float[] yScratch
    ) throws TornadoExecutionPlanException {
      TaskGraph taskGraph = taskGraph(key, xScratch, yScratch)
        .task(
          TASK_NAME,
          TornadoGemvKernels::gemvParallel,
          xScratch,
          key.weight(), key.weightOffset(),
          key.biasArg(), key.hasBiasFlag(),
          yScratch,
          key.in(), key.out0(), key.outCount(), key.work()
        )
        .transferToHost(EVERY_EXECUTION, (Object) yScratch);
      ImmutableTaskGraph snapshot = taskGraph.snapshot();
      TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot).withPreCompilation();
      return new GemvPlan(plan, xScratch, yScratch, key.rows(), key.in(), key.out0(),
        key.outCount());
    }

    private static TaskGraph taskGraph(
      final WeightKey key, final float[] xScratch, final float[] yScratch
    ) {
      TaskGraph taskGraph = new TaskGraph(GRAPH_NAME)
        .transferToDevice(FIRST_EXECUTION, (Object) key.weight());
      if (key.hasBias()) {
        taskGraph = taskGraph.transferToDevice(FIRST_EXECUTION, (Object) key.biasArg());
      }
      return taskGraph.transferToDevice(EVERY_EXECUTION, (Object) xScratch);
    }

    private static GemvPlan open(
      final TaskGraph taskGraph, final GridScheduler scheduler,
      final float[] xScratch, final float[] yScratch, final WeightKey key
    ) throws TornadoExecutionPlanException {
      ImmutableTaskGraph snapshot = taskGraph.snapshot();
      TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)
        .withPreCompilation()
        .withGridScheduler(scheduler);
      return new GemvPlan(plan, xScratch, yScratch, key.rows(), key.in(), key.out0(),
        key.outCount());
    }

    private static long localWorkSize(final int span) {
      long local = 256L;
      while (local > 1L && (span % local) != 0) {
        local >>= 1;
      }
      return Math.max(1L, local);
    }

    void execute(
      final float[] x, final int xOff,
      final float[] y, final int yOff, final int rowStride
    ) throws TornadoExecutionPlanException {
      System.arraycopy(x, xOff, this.xScratch, 0, this.rows * this.in);
      this.plan.execute();
      this.scatter(y, yOff, rowStride);
    }

    private void scatter(final float[] y, final int yOff, final int rowStride) {
      if (this.out0 == 0 && rowStride == this.outCount) {
        System.arraycopy(this.yScratch, 0, y, yOff, this.yScratch.length);
        return;
      }
      for (int r = 0; r < this.rows; r++) {
        System.arraycopy(
          this.yScratch, r * this.outCount, y, yOff + r * rowStride + this.out0, this.outCount);
      }
    }

    @Override
    public void close() throws TornadoExecutionPlanException {
      this.plan.close();
    }

    void closeQuietly() {
      try {
        this.close();
      } catch (TornadoExecutionPlanException failure) {
        Objects.requireNonNull(failure);
      }
    }
  }
}
