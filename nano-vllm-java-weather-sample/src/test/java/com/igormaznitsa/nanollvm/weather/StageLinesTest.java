package com.igormaznitsa.nanollvm.weather;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class StageLinesTest {

  private static final Place TALLINN = new Place("Tallinn", "Harjumaa", "Estonia", 59.437, 24.7536);

  @Test
  void cuesNameTheStageAndTheCity() {
    assertEquals("I am looking for location.", WeatherSample.lookingForLocation());
    assertEquals("I am getting weather info in Tallinn.", WeatherSample.gettingWeather(TALLINN));
  }

  @Test
  void missingCityFallsBackToTheFix() {
    Place fix = new Place("", "", "", 59.44, 24.75);

    assertEquals(
      "I am getting weather info in latitude 59.44, longitude 24.75.",
      WeatherSample.gettingWeather(fix));
  }
}
