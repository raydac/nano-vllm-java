package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * Launches {@link TornadoGemvKernels#gemv} on the default TornadoVM device.
 *
 * @since 1.4.0
 */
final class TornadoGemvExecutor {

  private static final int MAX_CACHED_PLANS = 32;
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
    private final int out1;

    private GemvSignature(
      final float[] x, final int xOff,
      final float[] w, final int wOff,
      final float[] bias,
      final float[] y, final int yOff,
      final int in, final int out0, final int out1
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
      this.out1 = out1;
    }

    static GemvSignature of(
      final float[] x, final int xOff,
      final float[] w, final int wOff,
      final float[] bias,
      final float[] y, final int yOff,
      final int in, final int out0, final int out1
    ) {
      return new GemvSignature(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
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

    boolean hasBias() {
      return this.bias != null;
    }

    float[] biasArg() {
      return this.hasBias() ? this.bias : TornadoGemvKernels.NO_BIAS;
    }

    int hasBiasFlag() {
      return this.hasBias() ? 1 : 0;
    }

    int out1() {
      return this.out1;
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
        && this.in == that.in && this.out0 == that.out0 && this.out1 == that.out1;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
        System.identityHashCode(this.x), this.xOff,
        System.identityHashCode(this.w), this.wOff,
        System.identityHashCode(this.bias),
        System.identityHashCode(this.y), this.yOff,
        this.in, this.out0, this.out1);
    }
  }

  private static final class GemvPlan implements AutoCloseable {

    private final TornadoExecutionPlan plan;

    private GemvPlan(final TornadoExecutionPlan plan) {
      this.plan = plan;
    }

    static GemvPlan compile(final GemvSignature signature) throws TornadoExecutionPlanException {
      final float[] biasArg = signature.biasArg();
      TaskGraph taskGraph = new TaskGraph("nanollvm-gemv")
        .transferToDevice(FIRST_EXECUTION, signature.w());
      if (signature.hasBias()) {
        taskGraph = taskGraph.transferToDevice(FIRST_EXECUTION, biasArg);
      }
      taskGraph = taskGraph
        .transferToDevice(EVERY_EXECUTION, signature.x())
        .task(
          "gemv",
          TornadoGemvKernels::gemv,
          signature.x(), signature.xOff(),
          signature.w(), signature.wOff(),
          biasArg, signature.hasBiasFlag(),
          signature.y(), signature.yOff(),
          signature.in(), signature.out0(), signature.out1()
        )
        .transferToHost(EVERY_EXECUTION, signature.y());
      final ImmutableTaskGraph snapshot = taskGraph.snapshot();
      final TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot).withPreCompilation();
      return new GemvPlan(plan);
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
