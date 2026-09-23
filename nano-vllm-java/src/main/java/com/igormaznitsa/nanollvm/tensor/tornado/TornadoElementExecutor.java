package com.igormaznitsa.nanollvm.tensor.tornado;

import static java.util.Locale.ROOT;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * Cached TornadoVM plans for elementwise and reduction kernels. One plan per operation and length.
 *
 * @since 1.5.0
 */
final class TornadoElementExecutor {

  private static final int MAX_CACHED_PLANS = 128;
  private static final Map<PlanKey, ElementPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<PlanKey, ElementPlan> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private TornadoElementExecutor() {
  }

  static float dot(
    final float[] left, final int leftOff, final float[] right, final int rightOff, final int n
  ) {
    return withPlan(Kind.DOT, n, plan -> {
      copy(left, leftOff, plan.left, n);
      copy(right, rightOff, plan.right, n);
      plan.sum[0] = 0f;
      plan.execute();
      return plan.sum[0];
    });
  }

  static float sumSquares(final float[] values, final int offset, final int n) {
    return withPlan(Kind.SUM_SQUARES, n, plan -> {
      copy(values, offset, plan.left, n);
      plan.sum[0] = 0f;
      plan.execute();
      return plan.sum[0];
    });
  }

  static void add(
    final float[] left, final int leftOff, final float[] right, final int rightOff,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.ADD, n, plan -> {
      copy(left, leftOff, plan.left, n);
      copy(right, rightOff, plan.right, n);
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  static void mul(
    final float[] left, final int leftOff, final float[] right, final int rightOff,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.MUL, n, plan -> {
      copy(left, leftOff, plan.left, n);
      copy(right, rightOff, plan.right, n);
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  static void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.SCALE, n, plan -> {
      copy(src, srcOff, plan.left, n);
      plan.scalar[0] = factor;
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  static void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    withPlan(Kind.AXPY, n, plan -> {
      copy(dst, dstOff, plan.left, n);
      copy(src, srcOff, plan.right, n);
      plan.scalar[0] = alpha;
      plan.execute();
      copy(plan.left, 0, dst, dstOff, n);
      return null;
    });
  }

  static void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int weightOff,
    final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.SCALE_ADD, n, plan -> {
      copy(src, srcOff, plan.left, n);
      copy(weight, weightOff, plan.right, n);
      plan.scalar[0] = scale;
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  static void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int weightOff,
    final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.SCALE_ADD_ONE_PLUS, n, plan -> {
      copy(src, srcOff, plan.left, n);
      copy(weight, weightOff, plan.right, n);
      plan.scalar[0] = scale;
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  static float addSumSquares(
    final float[] left, final int leftOff, final float[] right, final int rightOff,
    final float[] dst, final int dstOff, final int n
  ) {
    add(left, leftOff, right, rightOff, dst, dstOff, n);
    return sumSquares(dst, dstOff, n);
  }

  static void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    binary(Kind.SILU, gate, gateOff, up, upOff, dst, dstOff, n);
  }

  static void geluTanh(final float[] src, final int srcOff, final float[] dst, final int dstOff,
                       final int n) {
    unary(Kind.GELU, src, srcOff, dst, dstOff, n);
  }

  static void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    binary(Kind.GELU_MUL, gate, gateOff, up, upOff, dst, dstOff, n);
  }

  static void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(Kind.SOFTCAP, n, plan -> {
      copy(src, srcOff, plan.left, n);
      plan.scalar[0] = cap;
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  private static void unary(
    final Kind kind, final float[] src, final int srcOff, final float[] dst, final int dstOff,
    final int n
  ) {
    withPlan(kind, n, plan -> {
      copy(src, srcOff, plan.left, n);
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  private static void binary(
    final Kind kind,
    final float[] left, final int leftOff, final float[] right, final int rightOff,
    final float[] dst, final int dstOff, final int n
  ) {
    withPlan(kind, n, plan -> {
      copy(left, leftOff, plan.left, n);
      copy(right, rightOff, plan.right, n);
      plan.execute();
      copy(plan.out, 0, dst, dstOff, n);
      return null;
    });
  }

  private static void copy(final float[] src, final int srcOff, final float[] dst, final int n) {
    System.arraycopy(src, srcOff, dst, 0, n);
  }

  private static void copy(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    System.arraycopy(src, srcOff, dst, dstOff, n);
  }

  private static <T> T withPlan(final Kind kind, final int n, final PlanWork<T> work) {
    TornadoLaunchLock.lock();
    try {
      return work.apply(cached(kind, n));
    } catch (TornadoExecutionPlanException failure) {
      throw new IllegalStateException("TornadoVM kernel failed", failure);
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static ElementPlan cached(final Kind kind, final int n)
    throws TornadoExecutionPlanException {
    PlanKey key = new PlanKey(kind, n);
    ElementPlan plan = CACHED_PLANS.get(key);
    if (plan == null) {
      plan = ElementPlan.compile(kind, n);
      CACHED_PLANS.put(key, plan);
    }
    return plan;
  }

  private enum Kind {
    DOT,
    SUM_SQUARES,
    ADD,
    MUL,
    SCALE,
    AXPY,
    SCALE_ADD,
    SCALE_ADD_ONE_PLUS,
    SILU,
    GELU,
    GELU_MUL,
    SOFTCAP
  }

  @FunctionalInterface
  private interface PlanWork<T> {
    T apply(ElementPlan plan) throws TornadoExecutionPlanException;
  }

  private record PlanKey(Kind kind, int n) {
  }

  private static final class ElementPlan implements AutoCloseable {

    private final TornadoExecutionPlan plan;
    private final float[] left;
    private final float[] right;
    private final float[] out;
    private final float[] scalar;
    private final float[] sum;

    private ElementPlan(
      final TornadoExecutionPlan plan,
      final float[] left, final float[] right, final float[] out, final float[] scalar,
      final float[] sum
    ) {
      this.plan = plan;
      this.left = left;
      this.right = right;
      this.out = out;
      this.scalar = scalar;
      this.sum = sum;
    }

    static ElementPlan compile(final Kind kind, final int n) throws TornadoExecutionPlanException {
      float[] left = new float[n];
      float[] right = needsRight(kind) ? new float[n] : null;
      float[] out = needsOut(kind) ? new float[n] : null;
      float[] scalar = needsScalar(kind) ? new float[1] : null;
      float[] sum = needsSum(kind) ? new float[1] : null;
      String graphName = "nanollvm-" + kind.name().toLowerCase(ROOT);
      TaskGraph graph = new TaskGraph(graphName);
      graph = transferInputs(graph, left, right, scalar, sum);
      graph = attach(graph, kind, n, left, right, out, scalar, sum);
      if (sum != null) {
        graph = graph.transferToHost(EVERY_EXECUTION, (Object) sum);
      }
      if (out != null) {
        graph = graph.transferToHost(EVERY_EXECUTION, (Object) out);
      }
      if (kind == Kind.AXPY) {
        graph = graph.transferToHost(EVERY_EXECUTION, (Object) left);
      }
      ImmutableTaskGraph snapshot = graph.snapshot();
      TornadoExecutionPlan execution = new TornadoExecutionPlan(snapshot).withPreCompilation();
      return new ElementPlan(execution, left, right, out, scalar, sum);
    }

    private static TaskGraph transferInputs(
      final TaskGraph graph,
      final float[] left, final float[] right, final float[] scalar, final float[] sum
    ) {
      TaskGraph transferred = graph.transferToDevice(EVERY_EXECUTION, (Object) left);
      if (right != null) {
        transferred = transferred.transferToDevice(EVERY_EXECUTION, (Object) right);
      }
      if (scalar != null) {
        transferred = transferred.transferToDevice(EVERY_EXECUTION, (Object) scalar);
      }
      if (sum != null) {
        transferred = transferred.transferToDevice(EVERY_EXECUTION, (Object) sum);
      }
      return transferred;
    }

    private static TaskGraph attach(
      final TaskGraph graph, final Kind kind, final int n,
      final float[] left, final float[] right, final float[] out, final float[] scalar,
      final float[] sum
    ) {
      return switch (kind) {
        case DOT -> graph.task("run", TornadoElementKernels::dotProduct, left, right, sum);
        case SUM_SQUARES -> graph.task("run", TornadoElementKernels::sumSquares, left, sum);
        case ADD -> graph.task("run", TornadoElementKernels::add, left, right, out, n);
        case MUL -> graph.task("run", TornadoElementKernels::mul, left, right, out, n);
        case SCALE -> graph.task("run", TornadoElementKernels::scale, left, scalar, out, n);
        case AXPY -> graph.task("run", TornadoElementKernels::axpy, left, right, scalar, n);
        case SCALE_ADD ->
          graph.task("run", TornadoElementKernels::scaleAdd, left, right, scalar, out, n);
        case SCALE_ADD_ONE_PLUS ->
          graph.task("run", TornadoElementKernels::scaleAddOnePlus, left, right, scalar, out, n);
        case SILU -> graph.task("run", TornadoElementKernels::siluMul, left, right, out, n);
        case GELU -> graph.task("run", TornadoElementKernels::geluTanh, left, out, n);
        case GELU_MUL -> graph.task("run", TornadoElementKernels::geluTanhMul, left, right, out, n);
        case SOFTCAP -> graph.task("run", TornadoElementKernels::tanhSoftcap, left, scalar, out, n);
      };
    }

    private static boolean needsRight(final Kind kind) {
      return switch (kind) {
        case DOT, ADD, MUL, AXPY, SCALE_ADD, SCALE_ADD_ONE_PLUS, SILU, GELU_MUL -> true;
        case SUM_SQUARES, SCALE, GELU, SOFTCAP -> false;
      };
    }

    private static boolean needsOut(final Kind kind) {
      return switch (kind) {
        case ADD, MUL, SCALE, SCALE_ADD, SCALE_ADD_ONE_PLUS, SILU, GELU, GELU_MUL, SOFTCAP -> true;
        case DOT, SUM_SQUARES, AXPY -> false;
      };
    }

    private static boolean needsScalar(final Kind kind) {
      return switch (kind) {
        case SCALE, AXPY, SCALE_ADD, SCALE_ADD_ONE_PLUS, SOFTCAP -> true;
        case DOT, SUM_SQUARES, ADD, MUL, SILU, GELU, GELU_MUL -> false;
      };
    }

    private static boolean needsSum(final Kind kind) {
      return kind == Kind.DOT || kind == Kind.SUM_SQUARES;
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
