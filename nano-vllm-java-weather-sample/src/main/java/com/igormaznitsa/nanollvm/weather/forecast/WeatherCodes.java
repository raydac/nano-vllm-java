package com.igormaznitsa.nanollvm.weather.forecast;

import java.util.Locale;

/**
 * WMO weather interpretation codes used by Open-Meteo ({@code weather_code}).
 */
final class WeatherCodes {

  private WeatherCodes() {
  }

  static String english(final int code) {
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

  static String title(final int code) {
    String phrase = english(code);
    return phrase.substring(0, 1).toUpperCase(Locale.ROOT) + phrase.substring(1);
  }

  static String russian(final int code) {
    return switch (code) {
      case 0 -> "ясно";
      case 1 -> "преимущественно ясно";
      case 2 -> "переменная облачность";
      case 3 -> "пасмурно";
      case 45 -> "туман";
      case 48 -> "ледяной туман";
      case 51 -> "слабая морось";
      case 53 -> "морось";
      case 55 -> "сильная морось";
      case 56 -> "слабая ледяная морось";
      case 57 -> "ледяная морось";
      case 61 -> "небольшой дождь";
      case 63 -> "дождь";
      case 65 -> "сильный дождь";
      case 66 -> "небольшой ледяной дождь";
      case 67 -> "ледяной дождь";
      case 71 -> "небольшой снег";
      case 73 -> "снег";
      case 75 -> "сильный снег";
      case 77 -> "снежная крупа";
      case 80 -> "небольшой ливень";
      case 81 -> "ливень";
      case 82 -> "сильный ливень";
      case 85 -> "небольшой снегопад";
      case 86 -> "сильный снегопад";
      case 95 -> "гроза";
      case 96 -> "гроза с небольшим градом";
      case 99 -> "гроза с сильным градом";
      default -> "код погоды " + code;
    };
  }
}
