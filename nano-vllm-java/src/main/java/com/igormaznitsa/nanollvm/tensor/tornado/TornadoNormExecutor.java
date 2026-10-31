package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled RMSNorm plan per weight and shape. Every row of the tensor is one task, and the
 * gain stays on the device.
 *
 * @since 1.5.0
 */
final class TornadoNormExecutor {

  private static final int MAX_CACHED_PLANS = 256;

  private static final float[] NO_WEIGHT = new float[1];

  private static final Map<PlanKey, NormPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<PlanKey, NormPlan> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private TornadoNormExecutor() {
  }

  static void rmsNorm(
    final float[] src, final int srcOff,
    final float[] weight, final int weightOff,
    final float eps, final boolean onePlusWeight,
    final float[] dst, final int dstOff,
    final int rows, final int last
  ) {
    if (rows <= 0 || last <= 0) {
      return;
    }
    int mode = weight == null ? 0 : onePlusWeight ? 2 : 1;
    withPlan(Kind.RMS, weight, weightOff, eps, mode, rows, last, plan -> {
      System.arraycopy(src, srcOff, plan.left, 0, rows * last);
      plan.execute();
      System.arraycopy(plan.normed, 0, dst, dstOff, rows * last);
    });
  }

  static void addRmsNorm(
    final float[] left, final int leftOff,
    final float[] right, final int rightOff,
    final float[] weight, final int weightOff,
    final float eps, final boolean onePlusWeight,
    final float[] summed, final int summedOff,
    final float[] normed, final int normedOff,
    final int rows, final int last
  ) {
    if (rows <= 0 || last <= 0) {
      return;
    }
    int mode = weight == null ? 0 : onePlusWeight ? 2 : 1;
    withPlan(Kind.ADD_RMS, weight, weightOff, eps, mode, rows, last, plan -> {
      System.arraycopy(left, leftOff, plan.left, 0, rows * last);
      System.arraycopy(right, rightOff, plan.right, 0, rows * last);
      plan.execute();
      System.arraycopy(plan.summed, 0, summed, summedOff, rows * last);
      System.arraycopy(plan.normed, 0, normed, normedOff, rows * last);
    });
  }

  private static void withPlan(
    final Kind kind, final float[] weight, final int weightOff, final float eps, final int mode,
    final int rows, final int last, final PlanWork work
  ) {
    TornadoLaunchLock.lock();
    try {
      work.apply(cached(kind, weight, weightOff, eps, mode, rows, last));
    } catch (TornadoExecutionPlanException failure) {
      throw new IllegalStateException("TornadoVM RMSNorm failed", failure);
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static NormPlan cached(
    final Kind kind, final float[] weight, final int weightOff, final float eps, final int mode,
    final int rows, final int last
  ) throws TornadoExecutionPlanException {
    float[] identity = weight == null ? NO_WEIGHT : weight;
    PlanKey key = new PlanKey(kind, identity, weight == null ? 0 : weightOff, rows, last, mode,
      Float.floatToIntBits(eps));
    NormPlan plan = CACHED_PLANS.get(key);
    if (plan == null) {
      plan = NormPlan.compile(key, eps);
      CACHED_PLANS.put(key, plan);
    }
    return plan;
  }

  private enum Kind {
    RMS,
    ADD_RMS
  }

  @FunctionalInterface
  private interface PlanWork {
    void apply(NormPlan plan) throws TornadoExecutionPlanException;
  }

  private record PlanKey(Kind kind, float[] weight, int weightOffset, int rows, int last, int mode,
                         int epsBits) {

    @Override
    @SuppressWarnings("ReferenceEquality")
    public boolean equals(final Object other) {
      if (!(other instanceof PlanKey that)) {
        return false;
      }
      return this.kind == that.kind && this.weight == that.weight
        && this.weightOffset == that.weightOffset && this.rows == that.rows
        && this.last == that.last && this.mode == that.mode && this.epsBits == that.epsBits;
    }

    @Override
    public int hashCode() {
      return Objects.hash(this.kind, System.identityHashCode(this.weight), this.weightOffset,
        this.rows, this.last, this.mode, this.epsBits);
    }
  }

  private record NormPlan(TornadoExecutionPlan plan, float[] left, float[] right, float[] summed,
                          float[] normed) implements AutoCloseable {

    static NormPlan compile(final PlanKey key, final float eps)
      throws TornadoExecutionPlanException {
      int n = key.rows() * key.last();
      float[] left = new float[n];
      float[] right = key.kind() == Kind.ADD_RMS ? new float[n] : null;
      float[] summed = key.kind() == Kind.ADD_RMS ? new float[n] : null;
      float[] normed = new float[n];
      float[] gain = gain(key);
      float[] epsilon = new float[] {eps};
      String graphName = key.kind() == Kind.ADD_RMS ? "nanollvm-add-rms" : "nanollvm-rms";
      TaskGraph graph = new TaskGraph(graphName)
        .transferToDevice(FIRST_EXECUTION, gain, epsilon);
      graph = key.kind() == Kind.ADD_RMS
        ? graph.transferToDevice(EVERY_EXECUTION, left, right)
        : graph.transferToDevice(EVERY_EXECUTION, (Object) left);
      graph = key.kind() == Kind.ADD_RMS
        ? graph.task(
        "rms", TornadoNormKernels::addRmsNormRows,
        left, right, summed, gain, normed, epsilon, key.rows(), key.last())
        : graph.task(
        "rms", TornadoNormKernels::rmsNormRows,
        left, gain, normed, epsilon, key.rows(), key.last());
      if (summed != null) {
        graph = graph.transferToHost(EVERY_EXECUTION, summed, normed);
      } else {
        graph = graph.transferToHost(EVERY_EXECUTION, (Object) normed);
      }
      return new NormPlan(TornadoPlans.open(graph.snapshot()), left, right, summed, normed);
    }

    private static float[] gain(final PlanKey key) {
      float[] gain = new float[key.last()];
      if (key.mode() == 0) {
        Arrays.fill(gain, 1f);
        return gain;
      }
      System.arraycopy(key.weight(), key.weightOffset(), gain, 0, key.last());
      if (key.mode() == 2) {
        for (int i = 0; i < gain.length; i++) {
          gain[i] = 1f + gain[i];
        }
      }
      return gain;
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
