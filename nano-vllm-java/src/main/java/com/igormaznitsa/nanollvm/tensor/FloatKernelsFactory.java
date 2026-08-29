package com.igormaznitsa.nanollvm.tensor;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS;

import com.igormaznitsa.nanollvm.tensor.scalar.ScalarFloatKernels;
import com.igormaznitsa.nanollvm.tensor.vector.VectorFloatKernels;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.lang.reflect.Method;
import java.util.Locale;
import java.util.Optional;

/**
 * Builds a {@link FloatKernels} backend: TornadoVM when optional TornadoVM jars are present and a device
 * is reachable, otherwise Vector API when available, otherwise scalar.
 *
 * <p>Default selection honors {@code -Dnanollvm.kernels}:
 * <ul>
 *   <li>{@code auto} (default) — {@link #createBestAvailable()}</li>
 *   <li>{@code tornado} / {@code gpu} — require TornadoVM or fail</li>
 *   <li>{@code vector} / {@code simd} — require incubator Vector API or fail</li>
 *   <li>{@code scalar} / {@code plain} — force portable loops</li>
 * </ul>
 * {@link FloatKernels#get()} calls {@link #create()} once at class init.
 *
 * <p>TornadoVM support is compiled into this module but probed reflectively so stock JDK runs work
 * without {@code tornado.api} on the module path.
 *
 * @see FloatKernels
 */
public final class FloatKernelsFactory {

  private static final String VECTOR_KERNELS =
    "com.igormaznitsa.nanollvm.tensor.vector.VectorFloatKernels";

  private static final String TORNADO_PROVIDER =
    "com.igormaznitsa.nanollvm.tensor.tornado.TornadoFloatKernelsProvider";

  /**
   * Resolved once at class init — Vector incubator module + kernels class loadable.
   */
  private static final boolean VECTOR_API_AVAILABLE = probeVectorApi();

  /**
   * Resolved once at class init — optional Tornado add-on present with at least one device.
   */
  private static final boolean TORNADO_AVAILABLE = probeTornado();

  private FloatKernelsFactory() {
  }

  /**
   * Default selection: honors {@code -Dnanollvm.kernels}, else best available.
   */
  public static FloatKernels create() {
    return create(Optional.ofNullable(NanoLlvmProps.systemProperty(PROP_KERNELS)).orElse("auto"));
  }

  /**
   * Selects kernels for {@code mode}: {@code auto}, {@code tornado}/{@code gpu},
   * {@code vector}/{@code simd}, or {@code scalar}/{@code plain}.
   *
   * @param mode kernel mode
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
      case "tornado" -> requireTornado(createCpuFallback());
      case "auto" -> createBestAvailable();
      default -> throw new IllegalArgumentException(
        "Unknown -D" + PROP_KERNELS + "=" + mode + " (use auto|tornado|vector|scalar)"
      );
    };
  }

  /**
   * Prefer TornadoVM when the optional add-on is ready, else Vector API, else scalar.
   */
  public static FloatKernels createBestAvailable() {
    FloatKernels cpu = createCpuFallback();
    FloatKernels tornado = createTornadoIfAvailable(cpu);
    return tornado != null ? tornado : cpu;
  }

  /**
   * Cached result of the one-time Vector API probe at class initialization.
   */
  public static boolean isVectorApiAvailable() {
    return VECTOR_API_AVAILABLE;
  }

  private static FloatKernels createCpuFallback() {
    return VECTOR_API_AVAILABLE ? new VectorFloatKernels() : new ScalarFloatKernels();
  }

  private static FloatKernels requireTornado(final FloatKernels cpuFallback) {
    FloatKernels tornado = createTornadoIfAvailable(cpuFallback);
    if (tornado == null) {
      throw new IllegalStateException(
        "TornadoVM kernels requested (-D" + PROP_KERNELS
          + "=tornado) but TornadoVM is unavailable. Add optional tornado-api and tornado-runtime "
          +
          "on the module path and run on a TornadoVM-enabled JDK with at least one accelerator device."
      );
    }
    return tornado;
  }

  private static FloatKernels createTornadoIfAvailable(final FloatKernels cpuFallback) {
    if (!TORNADO_AVAILABLE) {
      return null;
    }
    try {
      Class<?> provider = Class.forName(TORNADO_PROVIDER);
      Method create = provider.getMethod("create", FloatKernels.class);
      Object result = create.invoke(null, cpuFallback);
      if (result instanceof FloatKernels kernels) {
        return kernels;
      }
    } catch (ReflectiveOperationException ignored) {
      return null;
    }
    return null;
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

  private static boolean probeTornado() {
    try {
      Class<?> provider = Class.forName(TORNADO_PROVIDER);
      Method isAvailable = provider.getMethod("isAvailable");
      Object result = isAvailable.invoke(null);
      return Boolean.TRUE.equals(result);
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
      case "gpu" -> "tornado";
      default -> m;
    };
  }
}
