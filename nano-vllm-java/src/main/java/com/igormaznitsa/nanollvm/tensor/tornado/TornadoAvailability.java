package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * One-time probe for a usable TornadoVM runtime and accelerator device.
 *
 * @since 1.4.0
 */
final class TornadoAvailability {

  private static final boolean READY = probe();

  private TornadoAvailability() {
  }

  static boolean isReady() {
    return READY;
  }

  private static boolean probe() {
    try {
      TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
      int backends = runtime.getNumBackends();
      for (int backendIndex = 0; backendIndex < backends; backendIndex++) {
        TornadoBackend backend = runtime.getBackend(backendIndex);
        if (backend.getNumDevices() > 0) {
          return true;
        }
      }
    } catch (Throwable ignored) {
      return false;
    }
    return false;
  }
}
