package com.igormaznitsa.nanollvm.weather;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class JsonValuesTest {

  @Test
  void readsANumberAfterAStringWithTheSameWord() {
    String json = """
      {"note":"see \\"latitude\\": 1","latitude":59.4,"longitude":24.7}""";

    assertEquals(Optional.of(59.4d), JsonValues.number(json, "latitude"));
    assertEquals(Optional.of("Europe/Tallinn"), JsonValues.text("""
      {"timezone":"Europe\\/Tallinn"}""", "timezone"));
  }

  @Test
  void readsTheNestedObjectRatherThanAShorterKey() {
    String json = """
      {"current_units":{"temperature_2m":"°C","weather_code":"wmo code"},\
      "current":{"temperature_2m":8.1,"weather_code":3}}""";

    String current = JsonValues.object(json, "current").orElseThrow();
    assertEquals(Optional.of(8.1d), JsonValues.number(current, "temperature_2m"));
    assertEquals(Optional.of(3d), JsonValues.number(current, "weather_code"));
    assertTrue(JsonValues.number(json, "temperature_2m").isEmpty());
  }
}
