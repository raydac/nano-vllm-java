package com.igormaznitsa.nanollvm.weather.place;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class PlaceLookupTest {

  @Test
  void ipwhoPlaceUsesCityRegionAndCountry() {
    Place place = IpPlace.parse("""
      {"success":true,"country":"Estonia","region":"Harjumaa","city":"Tallinn",\
      "latitude":59.437,"longitude":24.7536}""");

    assertEquals("Tallinn, Harjumaa, Estonia", place.where());
    assertEquals(
      "Tallinn, Harjumaa, Estonia (latitude 59.4370, longitude 24.7536)",
      place.forModel());
  }

  @Test
  void geojsAndIpapiShapesBothResolve() {
    Place geojs = IpPlace.parse("""
      {"timezone":"Europe\\/Tallinn","city":"Tallinn","region":"Harjumaa",\
      "country":"Estonia","latitude":"59.4381","longitude":"24.7369"}""");
    Place ipapi = IpPlace.parse("""
      {"city":"Tallinn","region":"Harjumaa","country":"EE","country_name":"Estonia",\
      "latitude":59.437,"longitude":24.754}""");

    assertEquals("Tallinn, Harjumaa, Estonia", geojs.where());
    assertEquals("Estonia", ipapi.country());
  }

  @Test
  void refusedLookupAndMissingCoordinatesFail() {
    assertThrows(IllegalArgumentException.class, () -> IpPlace.parse("""
      {"success":false,"message":"Invalid IP address"}"""));
    assertThrows(IllegalArgumentException.class,
      () -> new Place("X", "", "", 120d, 0d));
  }

  @Test
  void coordinatesOnlyPlaceStillNamesTheFix() {
    assertEquals(
      "latitude 59.44, longitude 24.75",
      new Place("", "  ", null, 59.437, 24.7536).where());
  }
}
