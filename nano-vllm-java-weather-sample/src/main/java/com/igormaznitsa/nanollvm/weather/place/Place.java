package com.igormaznitsa.nanollvm.weather.place;

import static java.util.stream.Collectors.joining;

import java.util.Locale;
import java.util.stream.Stream;

/**
 * City-level fix. Coordinates are the values the weather call uses.
 *
 * @param city      city name; empty when the lookup had none
 * @param region    region or state; empty when the lookup had none
 * @param country   country name; empty when the lookup had none
 * @param latitude  degrees north, {@code -90} through {@code 90}
 * @param longitude degrees east, {@code -180} through {@code 180}
 */
public record Place(String city, String region, String country, double latitude, double longitude) {

  public Place {
    city = city == null ? "" : city.strip();
    region = region == null ? "" : region.strip();
    country = country == null ? "" : country.strip();
    if (!Double.isFinite(latitude) || latitude < -90d || latitude > 90d) {
      throw new IllegalArgumentException("latitude out of range: " + latitude);
    }
    if (!Double.isFinite(longitude) || longitude < -180d || longitude > 180d) {
      throw new IllegalArgumentException("longitude out of range: " + longitude);
    }
  }

  public String where() {
    String named = Stream.of(this.city, this.region, this.country)
      .filter(part -> !part.isEmpty())
      .distinct()
      .collect(joining(", "));
    if (!named.isEmpty()) {
      return named;
    }
    return String.format(Locale.ROOT, "latitude %.2f, longitude %.2f", this.latitude,
      this.longitude);
  }

  public String forModel() {
    return this.where()
      + " (latitude " + String.format(Locale.ROOT, "%.4f", this.latitude)
      + ", longitude " + String.format(Locale.ROOT, "%.4f", this.longitude) + ")";
  }
}
