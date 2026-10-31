package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class BenchProfilerTest {

  private static void restore(final String key, final String previous) {
    if (previous == null) {
      System.clearProperty(key);
    } else {
      System.setProperty(key, previous);
    }
  }

  @Test
  void absentProfileDoesNotStart() {
    String previous = System.getProperty(BenchProfiler.PROP);
    System.clearProperty(BenchProfiler.PROP);
    try (BenchProfiler profiler = BenchProfiler.open()) {
      assertFalse(profiler.isActive());
    } finally {
      restore(BenchProfiler.PROP, previous);
    }
  }

  @Test
  void startCommandNamesTheEventAndInterval() {
    assertEquals(
      "start,event=cpu,interval=10000000",
      BenchProfiler.startCommand(" CPU ", 10_000_000L));
    assertEquals(BenchProfiler.DEFAULT_INTERVAL_NS, BenchProfiler.requireInterval(null));
    assertEquals(5_000L, BenchProfiler.requireInterval(" 5000 "));
  }

  @Test
  void unknownEventAndNonPositiveIntervalAreRejected() {
    assertThrows(IllegalArgumentException.class,
      () -> BenchProfiler.startCommand("cpu,file=/tmp/x", 1L));
    assertThrows(IllegalArgumentException.class, () -> BenchProfiler.requireInterval("0"));
    assertThrows(IllegalArgumentException.class, () -> BenchProfiler.requireInterval("nope"));
  }

  @Test
  void reportsSitBesideThePrefix() {
    Path base = Path.of("target", "bench-profile", "bench-scalar");
    assertTrue(BenchProfiler.withExtension(base, ".flat").endsWith("bench-scalar.flat"));
    assertEquals(
      base.resolveSibling("bench-scalar.collapsed"),
      BenchProfiler.withExtension(base, ".collapsed"));
  }
}
