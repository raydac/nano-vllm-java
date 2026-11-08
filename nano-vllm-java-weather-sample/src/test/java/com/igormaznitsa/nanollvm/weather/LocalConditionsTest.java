package com.igormaznitsa.nanollvm.weather;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LocalConditionsTest {

  @Test
  void weatherToolBeforeAPlaceDoesNotCallTheNetwork() {
    String result = new LocalConditions(new StepTimer()).weatherTool().call("{}");

    assertTrue(result.startsWith("error:"));
    assertFalse(result.contains("°C"));
  }
}
