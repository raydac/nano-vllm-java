package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled dense GEMV launch, either CLBlast or a TornadoVM kernel.
 *
 * @since 1.5.0
 */
interface TornadoGemvLaunch extends AutoCloseable {

  void execute(
    final float[] x, final int xOff, final float[] y, final int yOff, final int rowStride
  ) throws TornadoExecutionPlanException;

  @Override
  void close() throws TornadoExecutionPlanException;

  void closeQuietly();

  default boolean isClBlast() {
    return false;
  }
}
