package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class BenchTest {

  @Test
  void unsetLimitKeepsTheDefault() {
    String previous = System.getProperty(Bench.PROP_MAX_INPUT);
    System.clearProperty(Bench.PROP_MAX_INPUT);
    try {
      assertEquals(128, Bench.inclusiveLimit(Bench.PROP_MAX_INPUT, 128, 16));
    } finally {
      if (previous == null) {
        System.clearProperty(Bench.PROP_MAX_INPUT);
      } else {
        System.setProperty(Bench.PROP_MAX_INPUT, previous);
      }
    }
  }

  @Test
  void configuredLimitReplacesTheDefault() {
    System.setProperty(Bench.PROP_MAX_OUTPUT, "32");
    try {
      assertEquals(32, Bench.inclusiveLimit(Bench.PROP_MAX_OUTPUT, 64, 8));
    } finally {
      System.clearProperty(Bench.PROP_MAX_OUTPUT);
    }
  }

  @Test
  void limitBelowTheMinimumIsRejected() {
    System.setProperty(Bench.PROP_MAX_OUTPUT, "4");
    try {
      assertThrows(
        IllegalArgumentException.class,
        () -> Bench.inclusiveLimit(Bench.PROP_MAX_OUTPUT, 64, 8));
    } finally {
      System.clearProperty(Bench.PROP_MAX_OUTPUT);
    }
  }
}
