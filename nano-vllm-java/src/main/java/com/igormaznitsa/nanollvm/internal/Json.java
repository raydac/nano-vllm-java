package com.igormaznitsa.nanollvm.internal;

import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON reader for Hugging Face {@code config.json}, tokenizer sidecars, and similar
 * text catalogs. Caps input size with {@link ResourceLimits#maxJsonChars()}. Not a general
 * JSON library — numbers, strings, objects, arrays, and the usual literals only.
 */
public final class Json {

  private Json() {
  }

  /**
   * Parses a JSON value (object, array, string, number, {@code true}/{@code false}/{@code null}).
   *
   * @param text JSON text; must not be {@code null}
   * @return parsed tree ({@link Map}, {@link List}, {@link String}, {@link Number}, or
   * {@link Boolean}); {@code null} for JSON {@code null}
   * @throws IllegalArgumentException if {@code text} exceeds {@link ResourceLimits#maxJsonChars()}
   *                                  or is not valid JSON
   */
  public static Object parse(final String text) {
    ResourceLimits limits = ResourceLimits.current();
    if (text != null && text.length() > limits.maxJsonChars()) {
      throw new IllegalArgumentException(
        "JSON exceeds maxJsonChars (" + limits.maxJsonChars() + ")");
    }
    return new Parser(text, limits).parse();
  }

  /**
   * Parses a JSON object.
   *
   * @param text JSON object text; must not be {@code null}
   * @return the object map; never {@code null}
   * @throws IllegalArgumentException if the root is not an object, or parse fails
   */
  public static Map<String, Object> parseObject(final String text) {
    Object parsed = parse(text);
    if (!(parsed instanceof Map<?, ?> map)) {
      throw new IllegalArgumentException("expected JSON object");
    }
    return castObjectMap(map);
  }

  /**
   * {@link String#valueOf(Object)} of {@code value}, or {@code null} when {@code value} is
   * {@code null}.
   */
  public static String asString(final Object value) {
    return value == null ? null : String.valueOf(value);
  }

  /**
   * Integer from a parsed number or decimal string; {@code defaultValue} when {@code value} is
   * {@code null}.
   */
  public static int asInt(final Object value, final int defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.intValue();
    }
    return Integer.parseInt(value.toString());
  }

  /**
   * Long from a parsed number or decimal string; {@code defaultValue} when {@code value} is
   * {@code null}.
   */
  public static long asLong(final Object value, final long defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.longValue();
    }
    return Long.parseLong(value.toString());
  }

  /**
   * Float from a parsed number or decimal string; {@code defaultValue} when {@code value} is
   * {@code null}.
   */
  public static float asFloat(final Object value, final float defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.floatValue();
    }
    return Float.parseFloat(value.toString());
  }

  /**
   * Double from a parsed number or decimal string; {@code defaultValue} when {@code value} is
   * {@code null}.
   */
  public static double asDouble(final Object value, final double defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.doubleValue();
    }
    return Double.parseDouble(value.toString());
  }

  /**
   * Boolean from a parsed {@link Boolean} or {@link Boolean#parseBoolean(String)};
   * {@code defaultValue} when {@code value} is {@code null}.
   */
  public static boolean asBoolean(final Object value, final boolean defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Boolean b) {
      return b;
    }
    return Boolean.parseBoolean(value.toString());
  }

  /**
   * Casts a parsed object, or {@code null} when {@code value} is not a map.
   */
  @SuppressWarnings("unchecked")
  public static Map<String, Object> asObject(final Object value) {
    return value instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
  }

  /**
   * Casts a parsed array, or {@code null} when {@code value} is not a list.
   */
  @SuppressWarnings("unchecked")
  public static List<Object> asArray(final Object value) {
    return value instanceof List<?> l ? (List<Object>) l : null;
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> castObjectMap(final Map<?, ?> map) {
    return (Map<String, Object>) map;
  }

  private static final class Parser {
    private final String text;
    private final int maxDepth;
    private int pos;

    private Parser(final String text, final ResourceLimits limits) {
      this.text = text == null ? "null" : text;
      this.maxDepth = limits.maxJsonDepth();
    }

    private Object parse() {
      this.skipWhitespace();
      Object value = this.parseValue(0);
      this.skipWhitespace();
      if (!this.atEnd()) {
        throw this.error("unexpected trailing content");
      }
      return value;
    }

    private Object parseValue(final int depth) {
      if (depth > this.maxDepth) {
        throw this.error("JSON nesting exceeds maxJsonDepth (" + this.maxDepth + ")");
      }
      this.skipWhitespace();
      if (this.atEnd()) {
        throw this.error("unexpected end of input");
      }
      char ch = this.current();
      return switch (ch) {
        case '{' -> this.parseObject(depth);
        case '[' -> this.parseArray(depth);
        case '"' -> this.parseString();
        case 't' -> {
          this.expectWord("true");
          yield true;
        }
        case 'f' -> {
          this.expectWord("false");
          yield false;
        }
        case 'n' -> {
          this.expectWord("null");
          yield null;
        }
        default -> {
          if (ch == '-' || this.isDigit(ch)) {
            yield this.parseNumber();
          }
          throw this.error("unexpected token");
        }
      };
    }

    private Map<String, Object> parseObject(final int depth) {
      this.expect('{');
      this.skipWhitespace();
      Map<String, Object> result = new LinkedHashMap<>();
      if (this.tryConsume('}')) {
        return result;
      }
      while (true) {
        this.skipWhitespace();
        String key = this.parseString();
        this.skipWhitespace();
        this.expect(':');
        Object value = this.parseValue(depth + 1);
        result.put(key, value);
        this.skipWhitespace();
        if (this.tryConsume('}')) {
          return result;
        }
        this.expect(',');
      }
    }

    private List<Object> parseArray(final int depth) {
      this.expect('[');
      this.skipWhitespace();
      List<Object> result = new ArrayList<>();
      if (this.tryConsume(']')) {
        return result;
      }
      while (true) {
        result.add(this.parseValue(depth + 1));
        this.skipWhitespace();
        if (this.tryConsume(']')) {
          return result;
        }
        this.expect(',');
      }
    }

    private String parseString() {
      this.expect('"');
      StringBuilder result = new StringBuilder();
      while (!this.atEnd()) {
        char ch = this.current();
        this.pos++;
        if (ch == '"') {
          return result.toString();
        }
        if (ch == '\\') {
          result.append(this.parseEscape());
          continue;
        }
        if (ch < 0x20) {
          throw this.error("control char in string");
        }
        result.append(ch);
      }
      throw this.error("unterminated string");
    }

    private char parseEscape() {
      if (this.atEnd()) {
        throw this.error("unterminated escape");
      }
      char ch = this.current();
      this.pos++;
      return switch (ch) {
        case '"', '\\', '/' -> ch;
        case 'b' -> '\b';
        case 'f' -> '\f';
        case 'n' -> '\n';
        case 'r' -> '\r';
        case 't' -> '\t';
        case 'u' -> this.parseUnicode();
        default -> throw this.error("invalid escape sequence");
      };
    }

    private char parseUnicode() {
      if (this.pos + 4 > this.text.length()) {
        throw this.error("invalid unicode escape");
      }
      int code = 0;
      for (int i = 0; i < 4; i++) {
        char ch = this.text.charAt(this.pos++);
        int digit = Character.digit(ch, 16);
        if (digit < 0) {
          throw this.error("invalid unicode escape");
        }
        code = (code << 4) | digit;
      }
      return (char) code;
    }

    private Object parseNumber() {
      int start = this.pos;
      if (this.current() == '-') {
        this.pos++;
      }
      this.parseIntegerPart();
      boolean floating = false;
      if (!this.atEnd() && this.current() == '.') {
        floating = true;
        this.pos++;
        if (this.atEnd() || !this.isDigit(this.current())) {
          throw this.error("invalid number");
        }
        this.consumeDigits();
      }
      if (!this.atEnd() && (this.current() == 'e' || this.current() == 'E')) {
        floating = true;
        this.pos++;
        if (!this.atEnd() && (this.current() == '+' || this.current() == '-')) {
          this.pos++;
        }
        if (this.atEnd() || !this.isDigit(this.current())) {
          throw this.error("invalid exponent");
        }
        this.consumeDigits();
      }
      String number = this.text.substring(start, this.pos);
      try {
        if (floating) {
          return Double.parseDouble(number);
        }
        try {
          return Long.parseLong(number);
        } catch (NumberFormatException ignored) {
          return new BigInteger(number);
        }
      } catch (NumberFormatException ex) {
        throw this.error("invalid number");
      }
    }

    private void parseIntegerPart() {
      if (this.atEnd()) {
        throw this.error("invalid number");
      }
      if (this.current() == '0') {
        this.pos++;
        return;
      }
      if (!this.isDigit(this.current())) {
        throw this.error("invalid number");
      }
      this.consumeDigits();
    }

    private void consumeDigits() {
      while (!this.atEnd() && this.isDigit(this.current())) {
        this.pos++;
      }
    }

    private void expectWord(final String word) {
      if (this.pos + word.length() > this.text.length() || !this.text.startsWith(word, this.pos)) {
        throw this.error("unexpected token");
      }
      this.pos += word.length();
    }

    private void expect(final char expected) {
      this.skipWhitespace();
      if (this.atEnd() || this.current() != expected) {
        throw this.error("expected '" + expected + "'");
      }
      this.pos++;
    }

    private boolean tryConsume(final char token) {
      this.skipWhitespace();
      if (!this.atEnd() && this.current() == token) {
        this.pos++;
        return true;
      }
      return false;
    }

    private void skipWhitespace() {
      while (!this.atEnd()) {
        char ch = this.current();
        if (!Character.isWhitespace(ch)) {
          return;
        }
        this.pos++;
      }
    }

    private boolean isDigit(final char ch) {
      return ch >= '0' && ch <= '9';
    }

    private char current() {
      return this.text.charAt(this.pos);
    }

    private boolean atEnd() {
      return this.pos >= this.text.length();
    }

    private IllegalArgumentException error(final String message) {
      return new IllegalArgumentException(message + " at position " + this.pos);
    }
  }
}
