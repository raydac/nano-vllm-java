package com.igormaznitsa.nanollvm.llm;

import static java.util.Locale.ROOT;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Gemma 4 structured values. Strings are wrapped in the single token {@code <|"|>}. Object keys in
 * a tool call are bare identifiers; keys in a tool declaration are quoted.
 */
final class GemmaValues {

  static final String QUOTE = "<|\"|>";
  private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

  private final String text;
  private int pos;

  private GemmaValues(final String text) {
    this.text = text;
  }

  static Object parse(final String text) {
    GemmaValues parser = new GemmaValues(text.strip());
    Object value = parser.parseValue();
    parser.skipSpace();
    if (parser.pos != parser.text.length()) {
      throw new IllegalArgumentException("trailing gemma value");
    }
    return value;
  }

  static String format(final Object value, final boolean quoteKeys) {
    if (value == null) {
      return "null";
    }
    if (value instanceof String text) {
      return quote(text);
    }
    if (value instanceof Boolean || value instanceof Number) {
      return value.toString();
    }
    if (value instanceof Map<?, ?> map) {
      StringBuilder out = new StringBuilder();
      out.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> entry : map.entrySet()) {
        if (!first) {
          out.append(',');
        }
        first = false;
        String key = String.valueOf(entry.getKey());
        out.append(quoteKeys || !NAME.matcher(key).matches() ? quote(key) : key);
        out.append(':');
        Object child = entry.getValue();
        if (quoteKeys && "type".equals(key) && child instanceof String type) {
          out.append(quote(type.toUpperCase(ROOT)));
        } else {
          out.append(format(child, quoteKeys));
        }
      }
      out.append('}');
      return out.toString();
    }
    if (value instanceof List<?> items) {
      StringBuilder out = new StringBuilder();
      out.append('[');
      for (int i = 0; i < items.size(); i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(format(items.get(i), quoteKeys));
      }
      out.append(']');
      return out.toString();
    }
    return quote(String.valueOf(value));
  }

  static String quote(final String text) {
    return QUOTE + text.replace(QUOTE, "'") + QUOTE;
  }

  private Object parseValue() {
    this.skipSpace();
    if (this.pos >= this.text.length()) {
      throw new IllegalArgumentException("expected gemma value");
    }
    if (this.text.startsWith(QUOTE, this.pos)) {
      return this.parseQuoted();
    }
    char ch = this.text.charAt(this.pos);
    if (ch == '{') {
      return this.parseObject();
    }
    if (ch == '[') {
      return this.parseArray();
    }
    if (this.startsWith("true")) {
      this.pos += 4;
      return true;
    }
    if (this.startsWith("false")) {
      this.pos += 5;
      return false;
    }
    if (this.startsWith("null")) {
      this.pos += 4;
      return null;
    }
    return this.parseNumber();
  }

  private String parseQuoted() {
    this.pos += QUOTE.length();
    int end = this.text.indexOf(QUOTE, this.pos);
    if (end < 0) {
      throw new IllegalArgumentException("unterminated gemma string");
    }
    String value = this.text.substring(this.pos, end);
    this.pos = end + QUOTE.length();
    return value;
  }

  private Map<String, Object> parseObject() {
    this.pos++;
    Map<String, Object> object = new LinkedHashMap<>();
    this.skipSpace();
    if (this.consume('}')) {
      return object;
    }
    while (this.pos < this.text.length()) {
      String key = this.parseKey();
      this.skipSpace();
      this.require(':');
      object.put(key, this.parseValue());
      this.skipSpace();
      if (this.consume('}')) {
        return object;
      }
      this.require(',');
      this.skipSpace();
    }
    throw new IllegalArgumentException("unterminated gemma object");
  }

  private List<Object> parseArray() {
    this.pos++;
    List<Object> items = new ArrayList<>();
    this.skipSpace();
    if (this.consume(']')) {
      return items;
    }
    while (this.pos < this.text.length()) {
      items.add(this.parseValue());
      this.skipSpace();
      if (this.consume(']')) {
        return items;
      }
      this.require(',');
      this.skipSpace();
    }
    throw new IllegalArgumentException("unterminated gemma array");
  }

  private String parseKey() {
    this.skipSpace();
    if (this.text.startsWith(QUOTE, this.pos)) {
      return this.parseQuoted();
    }
    int start = this.pos;
    if (this.pos < this.text.length()) {
      char first = this.text.charAt(this.pos);
      if (Character.isLetter(first) || first == '_') {
        this.pos++;
        while (this.pos < this.text.length()) {
          char ch = this.text.charAt(this.pos);
          if (!Character.isLetterOrDigit(ch) && ch != '_') {
            break;
          }
          this.pos++;
        }
      }
    }
    if (this.pos == start) {
      throw new IllegalArgumentException("expected gemma key");
    }
    return this.text.substring(start, this.pos);
  }

  private Number parseNumber() {
    int start = this.pos;
    if (this.peek('-')) {
      this.pos++;
    }
    if (this.pos >= this.text.length() || !Character.isDigit(this.text.charAt(this.pos))) {
      throw new IllegalArgumentException("expected gemma number");
    }
    while (this.pos < this.text.length() && Character.isDigit(this.text.charAt(this.pos))) {
      this.pos++;
    }
    boolean floating = false;
    if (this.peek('.')) {
      floating = true;
      this.pos++;
      while (this.pos < this.text.length() && Character.isDigit(this.text.charAt(this.pos))) {
        this.pos++;
      }
    }
    String literal = this.text.substring(start, this.pos);
    if (floating) {
      return Double.valueOf(literal);
    }
    return Long.valueOf(literal);
  }

  private boolean startsWith(final String token) {
    if (!this.text.startsWith(token, this.pos)) {
      return false;
    }
    int end = this.pos + token.length();
    return end >= this.text.length() || !Character.isLetterOrDigit(this.text.charAt(end));
  }

  private void skipSpace() {
    while (this.pos < this.text.length() && Character.isWhitespace(this.text.charAt(this.pos))) {
      this.pos++;
    }
  }

  private boolean peek(final char ch) {
    return this.pos < this.text.length() && this.text.charAt(this.pos) == ch;
  }

  private boolean consume(final char ch) {
    if (!this.peek(ch)) {
      return false;
    }
    this.pos++;
    return true;
  }

  private void require(final char ch) {
    if (!this.consume(ch)) {
      throw new IllegalArgumentException("expected '" + ch + "'");
    }
  }
}
