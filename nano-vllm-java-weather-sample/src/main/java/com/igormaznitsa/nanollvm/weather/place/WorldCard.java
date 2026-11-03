package com.igormaznitsa.nanollvm.weather.place;

import static java.lang.Math.PI;
import static java.lang.Math.clamp;
import static java.lang.Math.floor;
import static java.lang.Math.sqrt;
import static java.lang.Math.toRadians;
import static java.util.Objects.requireNonNull;

import java.util.Arrays;
import java.util.Locale;

/**
 * 64-column world card. The coast is Kavrayskiy VII over the Natural Earth
 * 110m shoreline, and the mark is that same projection of the coordinates.
 */
public final class WorldCard {

  static final int WIDTH = 64;

  private static final int COLUMNS = WIDTH - 2;
  private static final int ROWS = 18;
  private static final double CENTRAL_MERIDIAN_DEGREES = 11d;
  private static final double HALF_WIDTH = 1.5d * PI * sqrt(1d / 3d);

  private static final String[] COAST = {
    "                     **  ***",
    "              ******* ******   *    * * ********",
    "       ************** *** **  *************************",
    "     * *  ***********       *********************** *",
    "          **********        ************************",
    "         ********           ****  ******************",
    "          ***  *          ************************",
    "            ***           *************  **  *** **",
    "               *****       ***********       ** ***",
    "               ********        *****           **** ** *",
    "                *******        ***** *            ****",
    "                 *****         **** *           *******",
    "                 ****           **              ** ***",
    "                 **                                 *  **",
    "                  **",
    "                                    *   *********",
    "            **********    *************************",
    "             ************************************"
  };

  private WorldCard() {
  }

  public static String of(final Place place) {
    requireNonNull(place, "place");
    char[][] coast = coast();
    coast[row(place.latitude())][column(place.longitude(), place.latitude())] = '@';

    StringBuilder card = new StringBuilder();
    card.append(rule()).append('\n');
    card.append(line(title(place))).append('\n');
    for (char[] coastLine : coast) {
      card.append(line(new String(coastLine))).append('\n');
    }
    card.append(rule());
    return card.toString();
  }

  static int column(final double longitude, final double latitude) {
    double lambda = longitudeRadians(longitude);
    double phi = phi(latitude);
    double disc = disc(phi);
    double x = disc <= 0d ? 0d : 1.5d * lambda * sqrt(disc);
    return clamp((int) floor((x + HALF_WIDTH) / (2d * HALF_WIDTH) * COLUMNS), 0, COLUMNS - 1);
  }

  static int row(final double latitude) {
    double scaled = (PI / 2d - phi(latitude)) / PI * ROWS;
    return clamp((int) floor(scaled), 0, ROWS - 1);
  }

  private static char[][] coast() {
    return Arrays.stream(COAST).map(WorldCard::padded).toArray(char[][]::new);
  }

  private static char[] padded(final String line) {
    char[] row = new char[COLUMNS];
    Arrays.fill(row, ' ');
    int count = Math.min(COLUMNS, line.length());
    line.getChars(0, count, row, 0);
    return row;
  }

  private static String title(final Place place) {
    String coords = coordinates(place);
    int room = COLUMNS - coords.length() - 1;
    String name = place.where();
    if (name.length() > room) {
      name = name.substring(0, Math.max(0, room));
    }
    int gap = COLUMNS - name.length() - coords.length();
    return name + " ".repeat(Math.max(0, gap)) + coords;
  }

  private static String coordinates(final Place place) {
    return String.format(
      Locale.ROOT,
      "%.2f°%s %.2f°%s",
      Math.abs(place.latitude()),
      place.latitude() >= 0d ? "N" : "S",
      Math.abs(place.longitude()),
      place.longitude() >= 0d ? "E" : "W");
  }

  private static String rule() {
    return "+" + "-".repeat(COLUMNS) + "+";
  }

  private static String line(final String content) {
    return "|" + content + "|";
  }

  private static double longitudeRadians(final double longitude) {
    double turned = toRadians(longitude - CENTRAL_MERIDIAN_DEGREES);
    double circle = 2d * PI;
    turned %= circle;
    if (turned > PI) {
      turned -= circle;
    }
    if (turned < -PI) {
      turned += circle;
    }
    return turned;
  }

  private static double phi(final double latitude) {
    return toRadians(clamp(latitude, -89.9d, 89.9d));
  }

  private static double disc(final double phi) {
    return 1d / 3d - (phi / PI) * (phi / PI);
  }
}
