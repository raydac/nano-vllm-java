package com.igormaznitsa.nanollvm.weather;

import static java.util.Objects.requireNonNull;

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

  private static String skyTitle(final int code) {
    String phrase = sky(code);
    return phrase.substring(0, 1).toUpperCase(Locale.ROOT) + phrase.substring(1);
  }

  private static String sky(final int code) {
    return switch (code) {
      case 0 -> "a clear sky";
      case 1 -> "a mainly clear sky";
      case 2 -> "a partly cloudy sky";
      case 3 -> "an overcast sky";
      case 45 -> "fog";
      case 48 -> "freezing fog";
      case 51 -> "light drizzle";
      case 53 -> "drizzle";
      case 55 -> "dense drizzle";
      case 56 -> "light freezing drizzle";
      case 57 -> "dense freezing drizzle";
      case 61 -> "light rain";
      case 63 -> "rain";
      case 65 -> "heavy rain";
      case 66 -> "light freezing rain";
      case 67 -> "heavy freezing rain";
      case 71 -> "light snow";
      case 73 -> "snow";
      case 75 -> "heavy snow";
      case 77 -> "snow grains";
      case 80 -> "light rain showers";
      case 81 -> "rain showers";
      case 82 -> "heavy rain showers";
      case 85 -> "light snow showers";
      case 86 -> "heavy snow showers";
      case 95 -> "a thunderstorm";
      case 96 -> "a thunderstorm and slight hail";
      case 99 -> "a thunderstorm and heavy hail";
      default -> "weather code " + code;
    };
  }

  public String logLine() {
    return String.format(
      Locale.ROOT,
      "%s: %.1f°C, %s",
      this.place.where(),
      tenths(this.celsius),
      skyTitle(this.weatherCode));
  }

  public String speech() {
    double rounded = tenths(this.celsius);
    double magnitude = Math.abs(rounded);
    String number = digits(magnitude);
    if (rounded < 0d) {
      number = "minus " + number;
    }
    String unit = new MessageFormat("{0,choice,0#degrees|1#degree|1<degrees}", Locale.ROOT)
      .format(new Object[] {magnitude});
    return "In %s, it is %s %s, with %s.".formatted(
      this.place.where(), number, unit, sky(this.weatherCode));
  }
}
