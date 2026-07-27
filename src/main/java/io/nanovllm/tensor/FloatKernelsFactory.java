package io.nanovllm.tensor;

import io.nanovllm.tensor.scalar.ScalarFloatKernels;
import io.nanovllm.tensor.vector.VectorFloatKernels;

import java.util.Locale;

/**
 * Builds a {@link FloatKernels} backend: Vector API when available, otherwise scalar.
 * Override with {@code -Dnanovllm.kernels=auto|vector|scalar}.
 */
public final class FloatKernelsFactory {

  private static final String VECTOR_KERNELS = "io.nanovllm.tensor.vector.VectorFloatKernels";

  /**
   * Resolved once at class init — Vector incubator module + kernels class loadable.
   */
  private static final boolean VECTOR_API_AVAILABLE = probeVectorApi();

  private FloatKernelsFactory() {
  }

  /**
   * Default selection: honors {@code -Dnanovllm.kernels}, else best available.
   */
  public static FloatKernels create() {
    return create(System.getProperty("nanovllm.kernels", "auto"));
  }

  /**
   * @param mode {@code auto} (default), {@code vector}/{@code simd}, or {@code scalar}/{@code plain}
   */
  public static FloatKernels create(String mode) {
    String selected = normalizeMode(mode);
    return switch (selected) {
      case "scalar" -> new ScalarFloatKernels();
      case "vector" -> {
        if (!VECTOR_API_AVAILABLE) {
          throw new IllegalStateException(
              "Vector API kernels requested (-Dnanovllm.kernels=vector) but Vector API is unavailable"
          );
        }
        yield new VectorFloatKernels();
      }
      case "auto" -> createBestAvailable();
      default -> throw new IllegalArgumentException(
          "Unknown -Dnanovllm.kernels=" + mode + " (use auto|vector|scalar)"
      );
    };
  }

  /**
   * Prefer Vector API kernels when the incubator module is present and loadable; otherwise scalar.
   */
  public static FloatKernels createBestAvailable() {
    return VECTOR_API_AVAILABLE ? new VectorFloatKernels() : new ScalarFloatKernels();
  }

  /**
   * Cached result of the one-time Vector API probe at class initialization.
   */
  public static boolean isVectorApiAvailable() {
    return VECTOR_API_AVAILABLE;
  }

  private static boolean probeVectorApi() {
    try {
      Class.forName("jdk.incubator.vector.FloatVector");
      Class.forName("jdk.incubator.vector.VectorSpecies");
      Class.forName(VECTOR_KERNELS);
      return true;
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static String normalizeMode(String mode) {
    if (mode == null || mode.isBlank()) {
      return "auto";
    }
    String m = mode.strip().toLowerCase(Locale.ROOT);
    return switch (m) {
      case "plain" -> "scalar";
      case "simd" -> "vector";
      default -> m;
    };
  }
}
