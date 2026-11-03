package com.igormaznitsa.nanollvm.weather.place;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class WorldCardTest {

  private static final Place TALLINN = new Place("Tallinn", "Harjumaa", "Estonia", 59.437, 24.7536);

  @Test
  void cardIs64ColumnsAndMarksTheCoordinates() {
    String[] lines = WorldCard.of(TALLINN).split("\n", -1);

    assertEquals(21, lines.length);
    for (String line : lines) {
      assertEquals(WorldCard.WIDTH, line.length());
    }
    assertEquals(1, WorldCard.of(TALLINN).chars().filter(ch -> ch == '@').count());
    assertEquals('@', lines[2 + WorldCard.row(TALLINN.latitude())].charAt(1 + WorldCard.column(
      TALLINN.longitude(),
      TALLINN.latitude())));
    assertTrue(lines[1].contains("Tallinn, Harjumaa, Estonia"));
    assertTrue(lines[1].contains("59.44°N"));
    assertTrue(lines[1].contains("24.75°E"));
  }

  @Test
  void projectionPutsEuropeRightOfAmericaAndAustraliaLower() {
    assertEquals(3, WorldCard.row(59.437));
    assertEquals(32, WorldCard.column(24.7536, 59.437));
    assertTrue(WorldCard.column(-74.01, 40.71) < WorldCard.column(24.7536, 59.437));
    assertTrue(WorldCard.column(-0.12, 51.5) < WorldCard.column(24.7536, 59.437));
    assertTrue(WorldCard.row(-33.87) > WorldCard.row(59.437));
    assertEquals(12, WorldCard.row(-33.87));
    assertEquals(53, WorldCard.column(151.21, -33.87));
  }

  @Test
  void southernAndWesternHemispheresUseThoseLetters() {
    String title = WorldCard.of(new Place("Sydney", "New South Wales", "Australia", -33.87, 151.21))
      .split("\n", -1)[1];

    assertTrue(title.contains("33.87°S"));
    assertTrue(title.contains("151.21°E"));
  }
}
