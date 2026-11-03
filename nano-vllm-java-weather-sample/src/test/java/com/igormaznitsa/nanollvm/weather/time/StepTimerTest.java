package com.igormaznitsa.nanollvm.weather.time;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class StepTimerTest {

  @Test
  void secondsUseAFixedDecimalPoint() {
    assertEquals("0.000 s", StepTimer.seconds(0L));
    assertEquals("1.500 s", StepTimer.seconds(1_500_000_000L));
  }

  @Test
  void measureReturnsTheStepValue() {
    assertEquals("Tallinn", new StepTimer().measure("step", () -> "Tallinn"));
  }
}