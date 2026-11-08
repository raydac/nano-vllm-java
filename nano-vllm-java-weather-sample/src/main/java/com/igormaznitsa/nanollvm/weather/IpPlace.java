package com.igormaznitsa.nanollvm.weather;

import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.net.URI;
import java.util.List;

/**
 * City of this computer from its public network address.
 *
 * <p>Three free HTTPS lookups, no API key: ipwho.is, then geojs, then ipapi.co.
 * MaxMind GeoIP2 is the usual Java library for this, and it needs a GeoLite2 database
 * downloaded with a MaxMind account, so the sample calls these endpoints with the JDK client.
 */
public final class IpPlace {

  private static final List<URI> ENDPOINTS = List.of(
    URI.create("https://ipwho.is/"),
    URI.create("https://get.geojs.io/v1/ip/geo.json"),
    URI.create("https://ipapi.co/json/")
  );

  private IpPlace() {
  }

  public static Place locate() {
    IllegalStateException failure = new IllegalStateException(
      "could not resolve this computer's location");
    for (URI endpoint : ENDPOINTS) {
      try {
        return parse(HttpsGet.body(endpoint));
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("location lookup was interrupted", ex);
      } catch (IOException | RuntimeException ex) {
        System.err.println("location lookup failed at " + endpoint.getHost());
        failure.addSuppressed(ex);
      }
    }
    throw failure;
  }

  static Place parse(final String json) {
    requireNonNull(json, "json");
    if (refused(json)) {
      throw new IllegalArgumentException("location lookup was refused");
    }
    double latitude = JsonValues.number(json, "latitude")
      .orElseThrow(() -> new IllegalArgumentException("location has no latitude"));
    double longitude = JsonValues.number(json, "longitude")
      .orElseThrow(() -> new IllegalArgumentException("location has no longitude"));
    return new Place(
      JsonValues.text(json, "city").orElse(""),
      JsonValues.text(json, "region").orElse(""),
      JsonValues.text(json, "country_name")
        .or(() -> JsonValues.text(json, "country"))
        .orElse(""),
      latitude,
      longitude);
  }

  private static boolean refused(final String json) {
    return JsonValues.value(json, "success").map("false"::equals).orElse(false);
  }
}
