package com.igormaznitsa.nanollvm.tensor.tornado;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_TORNADO_WEIGHT;

import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.util.Locale;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.memory.TornadoMemoryProvider;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Byte budget for one GEMV weight on the TornadoVM device.
 *
 * <p>{@link com.igormaznitsa.nanollvm.utils.NanoLlvmProps#PROP_TORNADO_WEIGHT}
 * ({@code -Dnanollvm.tornado.weight}) replaces the budget when it is set. Otherwise the budget
 * follows {@code -Dtornado.device.memory}: 32 MiB on the default 4 GiB heap, because the
 * decoder-layer graphs already fill that heap, and the spare bytes above 4 GiB when the heap is
 * larger. The device's reported global memory and its maximum single allocation cap the spare.
 * A matrix inside the budget stays resident for the current token count. A larger matrix is
 * copied through one reused tile of this many bytes.
 *
 * @see com.igormaznitsa.nanollvm.utils.NanoLlvmProps#PROP_TORNADO_WEIGHT
 * @since 1.5.0
 */
final class TornadoDeviceBudget {

  static final long DEFAULT_HEAP_BYTES = 4L * 1024 * 1024 * 1024;
  static final long COMPACT_TILE_BYTES = 32L * 1024 * 1024;

  private static final long MAX_WEIGHT_BYTES = 64L * 1024 * 1024 * 1024;

  private static volatile long cached = -1L;

  private TornadoDeviceBudget() {
  }

  /**
   * Budget for this process. {@code -Dnanollvm.tornado.weight} wins when it is set; otherwise the
   * default device supplies the heap, global memory, and maximum allocation to
   * {@link #weightBytes(long, long, long)}. The result is cached for the rest of the process.
   *
   * @return largest GEMV weight, in bytes, kept in one device buffer
   */
  static long weightBytes() {
    long known = cached;
    if (known >= 0L) {
      return known;
    }
    String configured = NanoLlvmProps.systemProperty(PROP_TORNADO_WEIGHT);
    long resolved = configured == null ? probe() : parseWeight(configured);
    cached = resolved;
    return resolved;
  }

  /**
   * Parses {@code -Dnanollvm.tornado.weight}.
   *
   * <p>A bare integer is a byte count. A suffix {@code K}, {@code M}, {@code G}, or {@code T},
   * with an optional {@code B} or {@code iB}, multiplies by successive powers of 1024.
   * {@code 32MB}, {@code 32MiB}, and {@code 32M} are the same size.
   *
   * @param raw property text; blank is rejected
   * @return byte count from 1 KiB through 64 GiB
   * @throws IllegalArgumentException when {@code raw} is blank, not a size, or outside that range
   */
  static long parseWeight(final String raw) {
    if (raw == null || raw.isBlank()) {
      throw new IllegalArgumentException(
        "-D" + PROP_TORNADO_WEIGHT + " must be a size such as 32MB");
    }
    String text = raw.strip().toUpperCase(Locale.ROOT);
    if (text.endsWith("IB") && text.length() > 3) {
      text = text.substring(0, text.length() - 2) + "B";
    }
    int unit = 0;
    String number = text;
    if (text.length() > 1 && text.endsWith("B")) {
      int shift = "KMGT".indexOf(text.charAt(text.length() - 2));
      if (shift >= 0) {
        unit = shift + 1;
        number = text.substring(0, text.length() - 2).strip();
      } else {
        number = text.substring(0, text.length() - 1).strip();
      }
    } else if (!text.isEmpty()) {
      int shift = "KMGT".indexOf(text.charAt(text.length() - 1));
      if (shift >= 0) {
        unit = shift + 1;
        number = text.substring(0, text.length() - 1).strip();
      }
    }
    long value;
    try {
      value = Long.parseLong(number);
    } catch (NumberFormatException failed) {
      throw new IllegalArgumentException(
        "-D" + PROP_TORNADO_WEIGHT + " must be a size such as 32MB, got " + raw, failed);
    }
    if (value < 0L) {
      throw new IllegalArgumentException(
        "-D" + PROP_TORNADO_WEIGHT + " must be from 1KB to 64GB, got " + raw);
    }
    long bytes = value;
    for (int step = 0; step < unit; step++) {
      if (bytes > Long.MAX_VALUE / 1024L) {
        throw new IllegalArgumentException(
          "-D" + PROP_TORNADO_WEIGHT + " must be from 1KB to 64GB, got " + raw);
      }
      bytes *= 1024L;
    }
    if (bytes < 1024L || bytes > MAX_WEIGHT_BYTES) {
      throw new IllegalArgumentException(
        "-D" + PROP_TORNADO_WEIGHT + " must be from 1KB to 64GB, got " + raw);
    }
    return bytes;
  }

  /**
   * Budget used when {@code -Dnanollvm.tornado.weight} is unset.
   *
   * @param heapBytes          TornadoVM heap ({@code -Dtornado.device.memory}); {@code 0} means 4 GiB
   * @param globalMemoryBytes  memory the device reports; {@code 0} leaves the heap uncapped by it
   * @param maxAllocationBytes largest single allocation the device allows; {@code 0} leaves the
   *                           budget uncapped by it
   * @return bytes one GEMV weight may occupy before it is streamed through a tile
   */
  static long weightBytes(
    final long heapBytes, final long globalMemoryBytes, final long maxAllocationBytes
  ) {
    long heap = heapBytes > 0L ? heapBytes : DEFAULT_HEAP_BYTES;
    long usable = globalMemoryBytes > 0L ? Math.min(heap, globalMemoryBytes) : heap;
    long allocation = maxAllocationBytes > 0L ? Math.min(usable, maxAllocationBytes) : usable;
    long spare = usable <= DEFAULT_HEAP_BYTES
      ? COMPACT_TILE_BYTES
      : Math.max(COMPACT_TILE_BYTES, usable - DEFAULT_HEAP_BYTES);
    return Math.min(spare, allocation);
  }

  private static long probe() {
    try {
      TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
      if (device == null) {
        return COMPACT_TILE_BYTES;
      }
      TornadoMemoryProvider memory = device.getMemoryProvider();
      long heap = memory == null ? 0L : memory.getHeapSize();
      return weightBytes(heap, device.getMaxGlobalMemory(), device.getMaxAllocMemory());
    } catch (Throwable ignored) {
      return COMPACT_TILE_BYTES;
    }
  }
}
