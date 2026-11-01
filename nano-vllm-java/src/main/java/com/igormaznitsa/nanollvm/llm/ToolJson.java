package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.internal.Json;
import java.util.Collection;
import java.util.Map;

/**
 * JSON text for tool advertisements and argument objects. Not a general writer: maps, lists,
 * strings, numbers, booleans, and null.
 */
final class ToolJson {

  private ToolJson() {
  }

  static String quote(final String text) {
    StringBuilder out = new StringBuilder(text.length() + 2);
    out.append('"');
    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);
      switch (ch) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (ch < 0x20) {
            out.append("\\u%04x".formatted((int) ch));
          } else {
            out.append(ch);
          }
        }
      }
    }
    out.append('"');
    return out.toString();
  }

  static String write(final Object value) {
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
        out.append(quote(String.valueOf(entry.getKey()))).append(':')
          .append(write(entry.getValue()));
      }
      out.append('}');
      return out.toString();
    }
    if (value instanceof Collection<?> items) {
      StringBuilder out = new StringBuilder();
      out.append('[');
      boolean first = true;
      for (Object item : items) {
        if (!first) {
          out.append(',');
        }
        first = false;
        out.append(write(item));
      }
      out.append(']');
      return out.toString();
    }
    return quote(String.valueOf(value));
  }

  /**
   * OpenAI function object the Qwen and LFM2 prompts embed. {@code parametersSchema} is spliced
   * as already-valid JSON.
   */
  static String openAiFunction(final ModelTool tool) {
    return "{\"type\":\"function\",\"function\":{\"name\":%s,\"description\":%s,\"parameters\":%s}}"
      .formatted(quote(tool.name()), quote(tool.description()), tool.parametersSchema());
  }

  /**
   * Bare function object LFM2 lists between its tool-list tokens.
   */
  static String bareFunction(final ModelTool tool) {
    return "{\"name\":%s,\"description\":%s,\"parameters\":%s}"
      .formatted(quote(tool.name()), quote(tool.description()), tool.parametersSchema());
  }

  /**
   * JSON object text for a tool body. A string that itself holds an object is unwrapped once.
   * Anything else becomes {@code {}}.
   */
  static String argumentsObject(final Object arguments) {
    if (arguments instanceof Map<?, ?> map) {
      return write(map);
    }
    if (arguments instanceof String text) {
      String stripped = text.strip();
      if (stripped.startsWith("{")) {
        try {
          return write(Json.parseObject(stripped));
        } catch (IllegalArgumentException ignored) {
          return "{}";
        }
      }
    }
    return "{}";
  }
}
