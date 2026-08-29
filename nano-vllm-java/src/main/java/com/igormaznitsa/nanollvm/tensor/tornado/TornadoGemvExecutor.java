package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;

import java.util.concurrent.locks.ReentrantLock;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * Launches {@link TornadoGemvKernels#gemv} on the default TornadoVM device.
 *
 * @since 1.3.1
 */
final class TornadoGemvExecutor {

  private static final ReentrantLock EXEC_LOCK = new ReentrantLock();

  private TornadoGemvExecutor() {
  }

  static void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    float[] biasArg = bias != null ? bias : TornadoGemvKernels.NO_BIAS;
    int hasBias = bias != null ? 1 : 0;

    TaskGraph taskGraph = new TaskGraph("nanollvm-gemv")
      .transferToDevice(EVERY_EXECUTION, x, w, biasArg, y)
      .task(
        "gemv",
        TornadoGemvKernels::gemv,
        x, xOff, w, wOff, biasArg, hasBias, y, yOff, in, out0, out1
      )
      .transferToHost(EVERY_EXECUTION, y);
    ImmutableTaskGraph snapshot = taskGraph.snapshot();

    EXEC_LOCK.lock();
    try {
      try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
        plan.withPreCompilation().execute();
      } catch (TornadoExecutionPlanException e) {
        throw new IllegalStateException("TornadoVM GEMV failed", e);
      }
    } finally {
      EXEC_LOCK.unlock();
    }
  }
}
