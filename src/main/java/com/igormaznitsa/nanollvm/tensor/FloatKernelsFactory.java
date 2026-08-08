package com.igormaznitsa.nanollvm.tensor;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS_LEGACY;

import com.igormaznitsa.nanollvm.tensor.scalar.ScalarFloatKernels;
import com.igormaznitsa.nanollvm.tensor.vector.VectorFloatKernels;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;

import java.util.Locale;
import java.util.Optional;

/**
 * Builds a {@link FloatKernels} backend: Vector API when available, otherwise scalar.
 *
 * <p>Default selection honors {@code -Dnanollvm.kernels} (legacy {@code nanovllm.kernels}):
 * <ul>
 *   <li>{@code auto} (default) — {@link #createBestAvailable()}</li>
 *   <li>{@code vector} / {@code simd} — require incubator Vector API or fail</li>
 *   <li>{@code scalar} / {@code plain} — force portable loops</li>
 * </ul>
 * {@link FloatKernels#get()} calls {@link #create()} once at class init.
 *
 * @see FloatKernels
 */
public final class FloatKernelsFactory {

  private static final String VECTOR_KERNELS =
      "com.igormaznitsa.nanollvm.tensor.vector.VectorFloatKernels";

  /**
   * Resolved once at class init — Vector incubator module + kernels class loadable.
   */
  private static final boolean VECTOR_API_AVAILABLE = probeVectorApi();

  private FloatKernelsFactory() {
  }

  /**
   * Default selection: honors {@code -Dnanollvm.kernels} (legacy {@code nanovllm.kernels}), else best available.
   */
  public static FloatKernels create() {
    return create(Optional.ofNullable(
      NanoLlvmProps.systemProperty(PROP_KERNELS, PROP_KERNELS_LEGACY)).orElse("auto"));
  }

  /**
   * @param mode {@code auto} (default), {@code vector}/{@code simd}, or {@code scalar}/{@code plain}
   */
  public static FloatKernels create(final String mode) {
    String selected = normalizeMode(mode);
    return switch (selected) {
      case "scalar" -> new ScalarFloatKernels();
      case "vector" -> {
        if (!VECTOR_API_AVAILABLE) {
          throw new IllegalStateException(
              "Vector API kernels requested (-D" + PROP_KERNELS
                  + "=vector) but Vector API is unavailable"
          );
        }
        yield new VectorFloatKernels();
      }
      case "auto" -> createBestAvailable();
      default -> throw new IllegalArgumentException(
          "Unknown -D" + PROP_KERNELS + "=" + mode + " (use auto|vector|scalar)"
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

  private static String normalizeMode(final String mode) {
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
