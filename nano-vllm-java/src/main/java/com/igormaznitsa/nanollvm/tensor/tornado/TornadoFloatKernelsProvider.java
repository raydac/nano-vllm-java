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
   * Builds TornadoVM kernels. GEMV, attention, and elementwise work run on the device.
   *
   * @return device kernels, or {@code null} when {@link #isAvailable()} is {@code false}
   * @since 1.4.0
   */
  public static FloatKernels create() {
    if (!TornadoAvailability.isReady()) {
      return null;
    }
    return new TornadoFloatKernels();
  }
}
