package com.igormaznitsa.nanollvm.weather;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ForecastTest {

  private static final Place TALLINN = new Place("Tallinn", "Harjumaa", "Estonia", 59.437, 24.7536);

  @Test
  void forecastIgnoresUnitLabelsAndSpeaksTheObservation() {
    Forecast forecast = Forecast.parse(TALLINN, """
      {"latitude":59.44,"longitude":24.75,\
      "current_units":{"temperature_2m":"°C","weather_code":"wmo code"},\
      "current":{"time":"2026-11-03T15:00","temperature_2m":8.1,"weather_code":3}}""");

    assertEquals("Tallinn, Harjumaa, Estonia: 8.1°C, An overcast sky", forecast.logLine());
    assertEquals(
      "In Tallinn, Harjumaa, Estonia, it is 8.1 degrees, with an overcast sky.",
      forecast.speech());
  }

  @Test
  void oneDegreeAndANegativeTemperature() {
    assertEquals(
      "In Tallinn, Harjumaa, Estonia, it is 1 degree, with a clear sky.",
      new Forecast(TALLINN, 1.04d, 0).speech());
    assertEquals(
      "In Tallinn, Harjumaa, Estonia, it is minus 2 degrees, with snow.",
      new Forecast(TALLINN, -2.04d, 73).speech());
  }
}
