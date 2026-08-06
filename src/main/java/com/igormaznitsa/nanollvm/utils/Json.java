package com.igormaznitsa.nanollvm.utils;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Json {

  private Json() {
  }

  public static Object parse(final String text) {
    return new Parser(text).parse();
  }

  public static Map<String, Object> parseObject(final String text) {
    Object parsed = parse(text);
    if (!(parsed instanceof Map<?, ?> map)) {
      throw new IllegalArgumentException("expected JSON object");
    }
    return castObjectMap(map);
  }

  public static String asString(final Object value) {
    return value == null ? null : String.valueOf(value);
  }

  public static int asInt(final Object value, final int defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.intValue();
    }
    return Integer.parseInt(value.toString());
  }

  public static long asLong(final Object value, final long defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.longValue();
    }
    return Long.parseLong(value.toString());
  }

  public static float asFloat(final Object value, final float defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.floatValue();
    }
    return Float.parseFloat(value.toString());
  }

  public static double asDouble(final Object value, final double defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.doubleValue();
    }
    return Double.parseDouble(value.toString());
  }

  public static boolean asBoolean(final Object value, final boolean defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Boolean b) {
      return b;
    }
    return Boolean.parseBoolean(value.toString());
  }

  @SuppressWarnings("unchecked")
  public static Map<String, Object> asObject(final Object value) {
    return value instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
  }

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
    private int pos;

    private Parser(final String text) {
      this.text = text == null ? "null" : text;
    }

    private Object parse() {
      this.skipWhitespace();
      Object value = this.parseValue();
      this.skipWhitespace();
      if (!this.atEnd()) {
        throw this.error("unexpected trailing content");
      }
      return value;
    }

    private Object parseValue() {
      this.skipWhitespace();
      if (this.atEnd()) {
        throw this.error("unexpected end of input");
      }
      char ch = this.current();
      return switch (ch) {
        case '{' -> this.parseObject();
        case '[' -> this.parseArray();
        case '"' -> this.parseString();
        case 't' -> this.parseTrue();
        case 'f' -> this.parseFalse();
        case 'n' -> this.parseNull();
        default -> {
          if (ch == '-' || this.isDigit(ch)) {
            yield this.parseNumber();
          }
          throw this.error("unexpected token");
        }
      };
    }

    private Map<String, Object> parseObject() {
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
        Object value = this.parseValue();
        result.put(key, value);
        this.skipWhitespace();
        if (this.tryConsume('}')) {
          return result;
        }
        this.expect(',');
      }
    }

    private List<Object> parseArray() {
      this.expect('[');
      this.skipWhitespace();
      List<Object> result = new ArrayList<>();
      if (this.tryConsume(']')) {
        return result;
      }
      while (true) {
        result.add(this.parseValue());
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

    private Boolean parseTrue() {
      this.expectWord("true");
      return Boolean.TRUE;
    }

    private Boolean parseFalse() {
      this.expectWord("false");
      return Boolean.FALSE;
    }

    private Object parseNull() {
      this.expectWord("null");
      return null;
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
