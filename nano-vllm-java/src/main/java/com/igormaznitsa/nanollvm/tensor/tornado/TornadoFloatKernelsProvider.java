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
   * Builds TornadoVM kernels. {@code cpuFallback} is accepted so callers keep a CPU backend ready;
   * the TornadoVM instance does not run that backend.
   *
   * @param cpuFallback Vector or scalar kernels used for non-GEMV work and small GEMV
   * @return hybrid kernels, or {@code null} when {@link #isAvailable()} is {@code false}
   * @since 1.4.0
   */
  public static FloatKernels create(final FloatKernels cpuFallback) {
    if (cpuFallback == null || !TornadoAvailability.isReady()) {
      return null;
    }
    return new TornadoFloatKernels();
  }
}
