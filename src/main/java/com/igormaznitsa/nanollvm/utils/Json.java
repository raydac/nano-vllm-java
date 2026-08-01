package com.igormaznitsa.nanollvm.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public final class Json {

  private static final Gson GSON = new GsonBuilder().create();
  private static final Type OBJECT_MAP = new TypeToken<Map<String, Object>>() {
  }.getType();

  private Json() {
  }

  public static Object parse(String text) {
    return GSON.fromJson(text, Object.class);
  }

  @SuppressWarnings("unchecked")
  public static Map<String, Object> parseObject(String text) {
    Map<String, Object> map = GSON.fromJson(text, OBJECT_MAP);
    if (map == null) {
      throw new IllegalArgumentException("expected JSON object");
    }
    return map;
  }

  public static String asString(Object value) {
    return value == null ? null : String.valueOf(value);
  }

  public static int asInt(Object value, int defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.intValue();
    }
    return Integer.parseInt(value.toString());
  }

  public static long asLong(Object value, long defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.longValue();
    }
    return Long.parseLong(value.toString());
  }

  public static float asFloat(Object value, float defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.floatValue();
    }
    return Float.parseFloat(value.toString());
  }

  public static double asDouble(Object value, double defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number n) {
      return n.doubleValue();
    }
    return Double.parseDouble(value.toString());
  }

  public static boolean asBoolean(Object value, boolean defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Boolean b) {
      return b;
    }
    return Boolean.parseBoolean(value.toString());
  }

  @SuppressWarnings("unchecked")
  public static Map<String, Object> asObject(Object value) {
    return value instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
  }

  @SuppressWarnings("unchecked")
  public static List<Object> asArray(Object value) {
    return value instanceof List<?> l ? (List<Object>) l : null;
  }
}
