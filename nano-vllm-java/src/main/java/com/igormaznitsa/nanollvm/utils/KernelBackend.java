package com.igormaznitsa.nanollvm.utils;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.Locale;
import java.util.Optional;

/**
 * Human-readable labels for the active float-kernel backend (scalar, Vector API, or TornadoVM).
 *
 * @since 1.4.0
 */
public final class KernelBackend {

  private KernelBackend() {
  }

  /**
   * Short label for console output: {@code Scalar Java}, {@code Vector API (SIMD)}, or
   * {@code TornadoVM}.
   *
   * @since 1.4.0
   */
  public static String label() {
    String name = VectorMath.backendInfo();
    if (name.startsWith("TornadoVM")) {
      return "TornadoVM";
    }
    if (name.contains("Vector API")) {
      return "Vector API (SIMD)";
    }
    return "Scalar Java";
  }

  /**
   * Active {@code -Dnanollvm.kernels} mode, defaulting to {@code auto}.
   *
   * @since 1.4.0
   */
  public static String mode() {
    return Optional.ofNullable(NanoLlvmProps.systemProperty(PROP_KERNELS)).orElse("auto")
      .strip().toLowerCase(Locale.ROOT);
  }

  /**
   * One-line summary for samples and logging.
   *
   * @since 1.4.0
   */
  public static String summaryLine() {
    return "Compute kernels: %s (mode=%s; override -D%s=auto|tornado|vector|scalar)."
      .formatted(label(), mode(), PROP_KERNELS);
  }

  /**
   * Full matmul runtime line (kernel name, tiling, thread count) as logged during model load.
   *
   * @since 1.4.0
   */
  public static String matmulDetail() {
    return MatmulRuntime.sequential().backendInfo();
  }
}
