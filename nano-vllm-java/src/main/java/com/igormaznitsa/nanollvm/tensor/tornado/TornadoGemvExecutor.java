package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled Tornado plan per weight matrix and token count. OpenCL uses CLBlast for more than
 * one token when {@code libclblast} runs; otherwise a device kernel. Activations are copied into a
 * plan-owned buffer so a new result array does not recompile the kernel or re-upload weights.
 *
 * @since 1.4.0
 */
final class TornadoGemvExecutor {

  private static final int MAX_CACHED_PLANS = 256;
  private static final String GRAPH_NAME = "nanollvm-gemv";
  private static final String TASK_NAME = "gemv";
  private static final String SCHEDULER_KEY = GRAPH_NAME + "." + TASK_NAME;
  private static final Map<WeightKey, TornadoGemvLaunch> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<WeightKey, TornadoGemvLaunch> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private TornadoGemvExecutor() {
  }

  static void releaseWeights(final float[]... weights) {
    if (weights == null || weights.length == 0) {
      return;
    }
    Set<float[]> drop = Collections.newSetFromMap(new IdentityHashMap<>());
    for (float[] weight : weights) {
      if (weight != null) {
        drop.add(weight);
      }
    }
    if (drop.isEmpty()) {
      return;
    }
    TornadoLaunchLock.lock();
    try {
      CACHED_PLANS.entrySet().removeIf(entry -> {
        if (!drop.contains(entry.getKey().weight())) {
          return false;
        }
        entry.getValue().closeQuietly();
        return true;
      });
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static TornadoGemvLaunch cached(final WeightKey key)
    throws TornadoExecutionPlanException {
    TornadoGemvLaunch plan = CACHED_PLANS.get(key);
    if (plan != null) {
      return plan;
    }
    TornadoGemvLaunch compiled = GemvPlan.compile(key);
    CACHED_PLANS.put(key, compiled);
    return compiled;
  }

  private static TornadoGemvLaunch replaceUnusableBlas(
    final WeightKey key, final TornadoGemvLaunch plan, final Exception failed
  ) throws TornadoExecutionPlanException {
    if (!plan.isClBlast()) {
      if (failed instanceof TornadoExecutionPlanException checked) {
        throw new IllegalStateException("TornadoVM GEMV failed", checked);
      }
      if (failed instanceof RuntimeException runtime) {
        throw runtime;
      }
      throw new IllegalStateException("TornadoVM GEMV failed", failed);
    }
    plan.closeQuietly();
    TornadoClBlastGemv.retire(failed);
    TornadoGemvLaunch kernel = GemvPlan.compileKernel(key);
    CACHED_PLANS.put(key, kernel);
    return kernel;
  }

  static int cachedPlanCount() {
    TornadoLaunchLock.lock();
    try {
      return CACHED_PLANS.size();
    } finally {
      TornadoLaunchLock.unlock();
    }
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
    TornadoLaunchLock.lock();
    try {
      TornadoGemvLaunch plan = cached(key);
      try {
        plan.execute(x, xOff, y, yOff, rowStride);
      } catch (RuntimeException | TornadoExecutionPlanException failed) {
        replaceUnusableBlas(key, plan, failed).execute(x, xOff, y, yOff, rowStride);
      }
    } catch (TornadoExecutionPlanException failed) {
      throw new IllegalStateException("TornadoVM GEMV failed", failed);
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private record WeightKey(float[] weight, int weightOffset, float[] bias, int in, int out0,
                           int outCount, int rows) {

    static WeightKey of(
      final float[] weight, final int weightOffset,
      final float[] bias,
      final int in, final int out0, final int outCount, final int rows
    ) {
      return new WeightKey(weight, weightOffset, bias, in, out0, outCount, rows);
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

  private record GemvPlan(TornadoExecutionPlan plan, float[] xScratch, float[] yScratch, int rows,
                          int in, int out0, int outCount) implements TornadoGemvLaunch {

    static TornadoGemvLaunch compile(final WeightKey key) throws TornadoExecutionPlanException {
      TornadoGemvLaunch blas = TornadoClBlastGemv.tryCompile(
        key.weight(), key.weightOffset(), key.bias(),
        key.in(), key.out0(), key.outCount(), key.rows());
      if (blas != null) {
        return blas;
      }
      return compileKernel(key);
    }

    static TornadoGemvLaunch compileKernel(final WeightKey key)
      throws TornadoExecutionPlanException {
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
      TornadoExecutionPlan plan = TornadoPlans.open(taskGraph.snapshot());
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
      TornadoExecutionPlan plan = TornadoPlans.open(taskGraph.snapshot())
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

    @Override
    public void execute(
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

    @Override
    public void closeQuietly() {
      try {
        this.close();
      } catch (TornadoExecutionPlanException failure) {
        Objects.requireNonNull(failure);
      }
    }
  }
}
