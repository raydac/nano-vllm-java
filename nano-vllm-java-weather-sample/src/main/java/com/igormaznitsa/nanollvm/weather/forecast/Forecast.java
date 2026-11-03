package com.igormaznitsa.nanollvm.weather.forecast;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.weather.http.HttpsGet;
import com.igormaznitsa.nanollvm.weather.json.JsonValues;
import com.igormaznitsa.nanollvm.weather.place.Place;

import java.io.IOException;
import java.net.URI;
import java.text.MessageFormat;
import java.util.Locale;

/**
 * Current temperature and sky at a {@link Place}, from Open-Meteo.
 *
 * <p>The Open-Meteo Java package on Maven Central decodes FlatBuffers only and does not
 * call the API. One JSON forecast is enough, and the data license is CC BY 4.0.
 *
 * @param place       place the coordinates were taken from
 * @param celsius     air temperature at 2 m
 * @param weatherCode WMO weather interpretation code
 */
public record Forecast(Place place, double celsius, int weatherCode) {

  public Forecast {
    requireNonNull(place, "place");
    if (!Double.isFinite(celsius)) {
      throw new IllegalArgumentException("temperature is not finite");
    }
  }

  public static Forecast at(final Place place) {
    requireNonNull(place, "place");
    URI uri = URI.create("https://api.open-meteo.com/v1/forecast?latitude="
      + String.format(Locale.ROOT, "%.4f", place.latitude())
      + "&longitude="
      + String.format(Locale.ROOT, "%.4f", place.longitude())
      + "&current=temperature_2m,weather_code&timezone=auto");
    try {
      return parse(place, HttpsGet.body(uri));
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("weather lookup was interrupted", ex);
    } catch (IOException | RuntimeException ex) {
      System.err.println("weather lookup failed at api.open-meteo.com");
      throw new IllegalStateException("could not read the weather", ex);
    }
  }

  static Forecast parse(final Place place, final String json) {
    requireNonNull(place, "place");
    requireNonNull(json, "json");
    String current = JsonValues.object(json, "current")
      .orElseThrow(
        () -> new IllegalArgumentException("weather response has no current observation"));
    double celsius = JsonValues.number(current, "temperature_2m")
      .orElseThrow(() -> new IllegalArgumentException("weather response has no temperature"));
    double code = JsonValues.number(current, "weather_code")
      .orElseThrow(() -> new IllegalArgumentException("weather response has no weather code"));
    return new Forecast(place, celsius, (int) Math.round(code));
  }

  private static double tenths(final double celsius) {
    return Math.round(celsius * 10.0d) / 10.0d;
  }

  private static String digits(final double magnitude) {
    if (Math.abs(magnitude - Math.rint(magnitude)) < 1e-9d) {
      return Long.toString(Math.round(magnitude));
    }
    return String.format(Locale.ROOT, "%.1f", magnitude);
  }

  public String logLine() {
    return String.format(
      Locale.ROOT,
      "%s: %.1f°C, %s",
      this.place.where(),
      tenths(this.celsius),
      WeatherCodes.title(this.weatherCode));
  }

  public String speech(final boolean english) {
    double rounded = tenths(this.celsius);
    double magnitude = Math.abs(rounded);
    String number = digits(magnitude);
    if (rounded < 0d) {
      number = (english ? "minus " : "минус ") + number;
    }
    if (!english) {
      return "В %s температура %s по Цельсию, %s.".formatted(
        this.place.where(), number, WeatherCodes.russian(this.weatherCode));
    }
    String unit = new MessageFormat("{0,choice,0#degrees|1#degree|1<degrees}", Locale.ROOT)
      .format(new Object[] {magnitude});
    return "In %s, it is %s %s, with %s.".formatted(
      this.place.where(), number, unit, WeatherCodes.english(this.weatherCode));
  }
}
