package com.igormaznitsa.nanollvm.weather;

import java.util.Optional;

/**
 * Reads one field from a JSON object. Enough for the location and weather payloads;
 * strings, numbers, and one nested object.
 */
public final class JsonValues {

  private JsonValues() {
  }

  public static Optional<String> text(final String json, final String key) {
    return value(json, key).flatMap(JsonValues::quoted);
  }

  public static Optional<Double> number(final String json, final String key) {
    return value(json, key).flatMap(JsonValues::asDouble);
  }

  public static Optional<String> object(final String json, final String key) {
    return value(json, key).filter(raw -> raw.startsWith("{"));
  }

  public static Optional<String> value(final String json, final String key) {
    if (json == null || key == null || key.isEmpty()) {
      return Optional.empty();
    }
    int at = valueAt(json, key);
    if (at < 0) {
      return Optional.empty();
    }
    String raw = readValue(json, at);
    return raw.isEmpty() ? Optional.empty() : Optional.of(raw);
  }

  private static Optional<String> quoted(final String raw) {
    if (raw.length() < 2 || raw.charAt(0) != '"') {
      return Optional.empty();
    }
    String text = unquote(raw).strip();
    return text.isEmpty() ? Optional.empty() : Optional.of(text);
  }

  private static Optional<Double> asDouble(final String raw) {
    String literal = raw.startsWith("\"") ? unquote(raw).strip() : raw.strip();
    if (literal.isEmpty() || "null".equals(literal)) {
      return Optional.empty();
    }
    try {
      double parsed = Double.parseDouble(literal);
      return Double.isFinite(parsed) ? Optional.of(parsed) : Optional.empty();
    } catch (NumberFormatException ex) {
      return Optional.empty();
    }
  }

  private static int valueAt(final String json, final String key) {
    String token = "\"" + key + "\"";
    int cursor = 0;
    while (cursor < json.length()) {
      if (json.charAt(cursor) != '"') {
        cursor++;
        continue;
      }
      if (json.startsWith(token, cursor)) {
        int afterKey = skipWs(json, cursor + token.length());
        if (afterKey < json.length() && json.charAt(afterKey) == ':') {
          return skipWs(json, afterKey + 1);
        }
      }
      cursor = endOfString(json, cursor);
    }
    return -1;
  }

  private static String readValue(final String json, final int start) {
    if (start >= json.length()) {
      return "";
    }
    char first = json.charAt(start);
    if (first == '"') {
      return json.substring(start, endOfString(json, start));
    }
    if (first == '{' || first == '[') {
      return readBalanced(json, start);
    }
    int end = start;
    while (end < json.length()) {
      char ch = json.charAt(end);
      if (ch == ',' || ch == '}' || ch == ']' || Character.isWhitespace(ch)) {
        break;
      }
      end++;
    }
    return json.substring(start, end);
  }

  private static String readBalanced(final String json, final int open) {
    int depth = 0;
    for (int i = open; i < json.length(); i++) {
      char ch = json.charAt(i);
      if (ch == '"') {
        i = endOfString(json, i) - 1;
        continue;
      }
      if (ch == '{' || ch == '[') {
        depth++;
      } else if (ch == '}' || ch == ']') {
        depth--;
        if (depth == 0) {
          return json.substring(open, i + 1);
        }
      }
    }
    return json.substring(open);
  }

  private static int endOfString(final String json, final int openingQuote) {
    for (int i = openingQuote + 1; i < json.length(); i++) {
      char ch = json.charAt(i);
      if (ch == '\\') {
        i++;
        continue;
      }
      if (ch == '"') {
        return i + 1;
      }
    }
    return json.length();
  }

  private static int skipWs(final String json, final int start) {
    int cursor = start;
    while (cursor < json.length() && Character.isWhitespace(json.charAt(cursor))) {
      cursor++;
    }
    return cursor;
  }

  private static String unquote(final String token) {
    StringBuilder value = new StringBuilder();
    for (int i = 1; i < token.length(); i++) {
      char ch = token.charAt(i);
      if (ch == '"') {
        return value.toString();
      }
      if (ch != '\\' || i + 1 >= token.length()) {
        value.append(ch);
        continue;
      }
      char escaped = token.charAt(++i);
      switch (escaped) {
        case 'n' -> value.append('\n');
        case 'r' -> value.append('\r');
        case 't' -> value.append('\t');
        case 'b' -> value.append('\b');
        case 'f' -> value.append('\f');
        case 'u' -> i = appendHex(token, i, value);
        default -> value.append(escaped);
      }
    }
    return value.toString();
  }

  private static int appendHex(final String token, final int uIndex, final StringBuilder value) {
    int hexStart = uIndex + 1;
    if (hexStart + 4 > token.length()) {
      value.append('u');
      return uIndex;
    }
    try {
      value.append((char) Integer.parseInt(token.substring(hexStart, hexStart + 4), 16));
    } catch (NumberFormatException ex) {
      value.append('u');
      return uIndex;
    }
    return hexStart + 3;
  }
}
