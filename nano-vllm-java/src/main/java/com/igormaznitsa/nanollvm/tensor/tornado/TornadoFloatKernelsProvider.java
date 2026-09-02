package com.igormaznitsa.nanollvm.tensor.tornado;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

/**
 * Entry point loaded reflectively by {@link com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory}
 * when TornadoVM is on the module path and a device is reachable.
 *
 * @since 1.4.0
 */
public final class TornadoFloatKernelsProvider {

  private TornadoFloatKernelsProvider() {
  }

  /**
   * Reports whether TornadoVM is on the module path and has at least one usable device.
   *
   * @return {@code true} when TornadoVM reports at least one usable device
   * @since 1.4.0
   */
  public static boolean isAvailable() {
    return TornadoAvailability.isReady();
  }

  /**
   * Wraps {@code cpuFallback} with TornadoVM GEMV offload when dimensions are large enough.
   *
   * @param cpuFallback Vector or scalar kernels used for non-GEMV work and small GEMV
   * @return hybrid kernels, or {@code null} when {@link #isAvailable()} is {@code false}
   * @since 1.4.0
   */
  public static FloatKernels create(final FloatKernels cpuFallback) {
    if (!TornadoAvailability.isReady()) {
      return null;
    }
    return new TornadoFloatKernels(cpuFallback);
  }
}
