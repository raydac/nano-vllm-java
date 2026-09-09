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
 * @since 1.4.0
 */
final class TornadoGemvExecutor {

  private static final int MAX_CACHED_PLANS = 32;
  private static final String GRAPH_NAME = "nanollvm-gemv";
  private static final String TASK_NAME = "gemv";
  private static final String SCHEDULER_KEY = GRAPH_NAME + "." + TASK_NAME;
  private static final ReentrantLock EXEC_LOCK = new ReentrantLock();
  private static final Map<GemvSignature, GemvPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<GemvSignature, GemvPlan> eldest) {
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
    final GemvSignature signature =
      GemvSignature.of(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
    EXEC_LOCK.lock();
    try {
      GemvPlan plan = CACHED_PLANS.get(signature);
      if (plan == null) {
        plan = GemvPlan.compile(signature);
        CACHED_PLANS.put(signature, plan);
      }
      plan.execute();
    } catch (TornadoExecutionPlanException e) {
      throw new IllegalStateException("TornadoVM GEMV failed", e);
    } finally {
      EXEC_LOCK.unlock();
    }
  }

  private static final class GemvSignature {

    private final float[] x;
    private final int xOff;
    private final float[] w;
    private final int wOff;
    private final float[] bias;
    private final float[] y;
    private final int yOff;
    private final int in;
    private final int out0;
    private final int outCount;

    private GemvSignature(
      final float[] x, final int xOff,
      final float[] w, final int wOff,
      final float[] bias,
      final float[] y, final int yOff,
      final int in, final int out0, final int outCount
    ) {
      this.x = x;
      this.xOff = xOff;
      this.w = w;
      this.wOff = wOff;
      this.bias = bias;
      this.y = y;
      this.yOff = yOff;
      this.in = in;
      this.out0 = out0;
      this.outCount = outCount;
    }

    static GemvSignature of(
      final float[] x, final int xOff,
      final float[] w, final int wOff,
      final float[] bias,
      final float[] y, final int yOff,
      final int in, final int out0, final int out1
    ) {
      return new GemvSignature(x, xOff, w, wOff, bias, y, yOff, in, out0, out1 - out0);
    }

    @SuppressWarnings("ReferenceEquality")
    private static boolean buffersIdentical(final float[] left, final float[] right) {
      return left == right;
    }

    float[] x() {
      return this.x;
    }

    int xOff() {
      return this.xOff;
    }

    float[] w() {
      return this.w;
    }

    int wOff() {
      return this.wOff;
    }

    float[] y() {
      return this.y;
    }

    int yOff() {
      return this.yOff;
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
    public boolean equals(final Object other) {
      if (!(other instanceof GemvSignature that)) {
        return false;
      }
      return buffersIdentical(this.x, that.x) && this.xOff == that.xOff
        && buffersIdentical(this.w, that.w) && this.wOff == that.wOff
        && buffersIdentical(this.bias, that.bias)
        && buffersIdentical(this.y, that.y) && this.yOff == that.yOff
        && this.in == that.in && this.out0 == that.out0 && this.outCount == that.outCount;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
        System.identityHashCode(this.x), this.xOff,
        System.identityHashCode(this.w), this.wOff,
        System.identityHashCode(this.bias),
        System.identityHashCode(this.y), this.yOff,
        this.in, this.out0, this.outCount);
    }
  }

  private static final class GemvPlan implements AutoCloseable {

    private final TornadoExecutionPlan plan;

    private GemvPlan(final TornadoExecutionPlan plan) {
      this.plan = plan;
    }

    static GemvPlan compile(final GemvSignature signature) throws TornadoExecutionPlanException {
      try {
        return compileKernelApi(signature);
      } catch (Throwable kernelApiFailed) {
        return compileLoopParallel(signature);
      }
    }

    private static GemvPlan compileKernelApi(final GemvSignature signature)
      throws TornadoExecutionPlanException {
      final float[] biasArg = signature.biasArg();
      final KernelContext context = new KernelContext();
      final int outCount = signature.outCount();
      final WorkerGrid1D worker = new WorkerGrid1D(outCount);
      worker.setLocalWork(localWorkSize(outCount), 1, 1);
      final GridScheduler scheduler = new GridScheduler(SCHEDULER_KEY, worker);

      TaskGraph taskGraph = new TaskGraph(GRAPH_NAME)
        .transferToDevice(FIRST_EXECUTION, (Object) signature.w());
      if (signature.hasBias()) {
        taskGraph = taskGraph.transferToDevice(FIRST_EXECUTION, (Object) biasArg);
      }
      taskGraph = taskGraph
        .transferToDevice(EVERY_EXECUTION, (Object) signature.x())
        .task(
          TASK_NAME,
          TornadoGemvKernels::gemvKernel,
          context,
          signature.x(), signature.xOff(),
          signature.w(), signature.wOff(),
          biasArg, signature.hasBiasFlag(),
          signature.y(), signature.yOff(),
          signature.in(), signature.out0(), outCount
        )
        .transferToHost(EVERY_EXECUTION, (Object) signature.y());

      final ImmutableTaskGraph snapshot = taskGraph.snapshot();
      final TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)
        .withPreCompilation()
        .withGridScheduler(scheduler);
      return new GemvPlan(plan);
    }

    private static GemvPlan compileLoopParallel(final GemvSignature signature)
      throws TornadoExecutionPlanException {
      final float[] biasArg = signature.biasArg();
      TaskGraph taskGraph = new TaskGraph(GRAPH_NAME)
        .transferToDevice(FIRST_EXECUTION, (Object) signature.w());
      if (signature.hasBias()) {
        taskGraph = taskGraph.transferToDevice(FIRST_EXECUTION, (Object) biasArg);
      }
      taskGraph = taskGraph
        .transferToDevice(EVERY_EXECUTION, (Object) signature.x())
        .task(
          TASK_NAME,
          TornadoGemvKernels::gemvParallel,
          signature.x(), signature.xOff(),
          signature.w(), signature.wOff(),
          biasArg, signature.hasBiasFlag(),
          signature.y(), signature.yOff(),
          signature.in(), signature.out0(), signature.outCount()
        )
        .transferToHost(EVERY_EXECUTION, (Object) signature.y());
      final ImmutableTaskGraph snapshot = taskGraph.snapshot();
      final TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot).withPreCompilation();
      return new GemvPlan(plan);
    }

    private static long localWorkSize(final int outCount) {
      long local = 256L;
      while (local > 1L && (outCount % local) != 0) {
        local >>= 1;
      }
      return Math.max(1L, local);
    }

    void execute() throws TornadoExecutionPlanException {
      this.plan.execute();
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
