package com.igormaznitsa.nanollvm.tensor.tornado;

import static java.lang.System.Logger.Level.WARNING;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;
import static uk.ac.manchester.tornado.clblast.enums.CLBlastLayout.ROW_MAJOR;
import static uk.ac.manchester.tornado.clblast.enums.CLBlastTranspose.NO;
import static uk.ac.manchester.tornado.clblast.enums.CLBlastTranspose.YES;

import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.clblast.CLBlast;

/**
 * Multi-token dense GEMV through TornadoVM 8 CLBlast {@code SGEMM}. One token stays on the device
 * kernel: CLBlast {@code SGEMV} can return non-finite values on some OpenCL CPU devices. The
 * weight slice stays on the device. Missing {@code libclblast} keeps the kernel plan.
 *
 * @since 1.5.0
 */
final class TornadoClBlastGemv {

  private static final System.Logger LOG = System.getLogger(TornadoClBlastGemv.class.getName());

  private static final String GRAPH_NAME = "nanollvm-clblast-gemv";

  private static final String TASK_NAME = "gemv";

  private static final AtomicInteger LAUNCHES = new AtomicInteger();

  private static volatile Boolean installed;

  private TornadoClBlastGemv() {
  }

  static boolean isInstalled() {
    Boolean ready = installed;
    if (ready != null) {
      return ready;
    }
    boolean probed = Boolean.parseBoolean(System.getProperty("nanollvm.tornado.clblast", "true"))
      && nativeLibraryPresent();
    installed = probed;
    return probed;
  }

  static int launchCount() {
    return LAUNCHES.get();
  }

  static void retire(final Throwable failed) {
    if (Boolean.FALSE.equals(installed)) {
      return;
    }
    installed = Boolean.FALSE;
    LOG.log(WARNING, "CLBlast dense GEMV is unavailable; TornadoVM keeps the device kernel",
      failed);
  }

  private static boolean nativeLibraryPresent() {
    try {
      Class<?> provider = Class.forName(
        "uk.ac.manchester.tornado.clblast.provider.CLBlastLibraryProvider");
      Object ready = provider.getMethod("isAvailable").invoke(null);
      return ready instanceof Boolean available && available;
    } catch (ReflectiveOperationException | LinkageError failed) {
      return false;
    }
  }

  static TornadoGemvLaunch tryCompile(
    final float[] weight, final int weightOffset, final float[] bias,
    final int in, final int out0, final int outCount, final int rows
  ) {
    if (!isInstalled() || in <= 0 || outCount <= 0 || rows <= 1) {
      return null;
    }
    try {
      int slice = Math.multiplyExact(outCount, in);
      int start = Math.addExact(weightOffset, Math.multiplyExact(out0, in));
      FloatArray matrix = FloatArray.fromFloatBuffer(FloatBuffer.wrap(weight, start, slice));
      FloatArray activation = new FloatArray(Math.multiplyExact(rows, in));
      FloatArray result = new FloatArray(Math.multiplyExact(rows, outCount));
      TornadoExecutionPlan plan = openGemm(matrix, activation, result, rows, outCount, in);
      return new BlasPlan(plan, activation, result, new float[rows * outCount], bias, rows, in,
        out0, outCount);
    } catch (LinkageError | RuntimeException | TornadoExecutionPlanException failed) {
      installed = Boolean.FALSE;
      LOG.log(WARNING, "CLBlast dense GEMV is unavailable; TornadoVM keeps the device kernel",
        failed);
      return null;
    }
  }

  private static TornadoExecutionPlan openGemm(
    final FloatArray matrix, final FloatArray activation, final FloatArray result,
    final int rows, final int outCount, final int in
  ) throws TornadoExecutionPlanException {
    TaskGraph graph = residentGraph(matrix, activation)
      .libraryTask(
        TASK_NAME,
        CLBlast::sgemm,
        ROW_MAJOR.value(),
        NO.value(),
        YES.value(),
        rows,
        outCount,
        in,
        1.0f,
        activation,
        in,
        matrix,
        in,
        0.0f,
        result,
        outCount)
      .transferToHost(EVERY_EXECUTION, (Object) result);
    return TornadoPlans.open(graph.snapshot());
  }

  private static TaskGraph residentGraph(final FloatArray matrix, final FloatArray activation) {
    return new TaskGraph(GRAPH_NAME)
      .transferToDevice(FIRST_EXECUTION, (Object) matrix)
      .transferToDevice(EVERY_EXECUTION, (Object) activation);
  }

  private static void copyFromHeap(
    final FloatArray dst, final float[] src, final int srcOff, final int count
  ) {
    for (int index = 0; index < count; index++) {
      dst.set(index, src[srcOff + index]);
    }
  }

  private static boolean hasNonFinite(final float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return true;
      }
    }
    return false;
  }

  private static void copyToHeap(final FloatArray src, final float[] dst) {
    for (int index = 0; index < dst.length; index++) {
      dst[index] = src.get(index);
    }
  }

  private record BlasPlan(
    TornadoExecutionPlan plan,
    FloatArray activation,
    FloatArray result,
    float[] yScratch,
    float[] bias,
    int rows,
    int in,
    int out0,
    int outCount
  ) implements TornadoGemvLaunch {

    @Override
    public void execute(
      final float[] x, final int xOff, final float[] y, final int yOff, final int rowStride
    ) throws TornadoExecutionPlanException {
      copyFromHeap(this.activation, x, xOff, this.rows * this.in);
      this.plan.execute();
      copyToHeap(this.result, this.yScratch);
      if (hasNonFinite(this.yScratch)) {
        throw new IllegalStateException("CLBlast GEMV produced a non-finite value");
      }
      this.addBias();
      this.scatter(y, yOff, rowStride);
      LAUNCHES.incrementAndGet();
    }

    @Override
    public boolean isClBlast() {
      return true;
    }

    private void addBias() {
      if (this.bias == null) {
        return;
      }
      for (int row = 0; row < this.rows; row++) {
        int base = row * this.outCount;
        for (int col = 0; col < this.outCount; col++) {
          this.yScratch[base + col] += this.bias[this.out0 + col];
        }
      }
    }

    private void scatter(final float[] y, final int yOff, final int rowStride) {
      if (this.out0 == 0 && rowStride == this.outCount) {
        System.arraycopy(this.yScratch, 0, y, yOff, this.yScratch.length);
        return;
      }
      for (int row = 0; row < this.rows; row++) {
        System.arraycopy(
          this.yScratch, row * this.outCount, y, yOff + row * rowStride + this.out0,
          this.outCount);
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
