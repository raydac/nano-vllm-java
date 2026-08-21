package com.igormaznitsa.nanollvm.tokenizer;

import static java.text.Normalizer.Form.NFC;
import static java.text.Normalizer.Form.NFD;
import static java.text.Normalizer.Form.NFKC;
import static java.text.Normalizer.Form.NFKD;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.Json;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * Hugging Face / SentencePiece normalizer chain.
 *
 * @since 1.2.0
 */
final class TextNormalizer {

  private static final TextNormalizer IDENTITY = new TextNormalizer(List.of());

  private final List<UnaryOperator<String>> steps;

  private TextNormalizer(final List<UnaryOperator<String>> steps) {
    this.steps = List.copyOf(steps);
  }

  static TextNormalizer identity() {
    return IDENTITY;
  }

  static TextNormalizer lowercase() {
    return new TextNormalizer(List.of(text -> text.toLowerCase(ROOT)));
  }

  static TextNormalizer of(final UnaryOperator<String> step) {
    return new TextNormalizer(List.of(requireNonNull(step, "step")));
  }

  static TextNormalizer fromHf(final Object node) {
    if (node == null) {
      return IDENTITY;
    }
    List<UnaryOperator<String>> steps = new ArrayList<>();
    collect(node, steps);
    return steps.isEmpty() ? IDENTITY : new TextNormalizer(steps);
  }

  private static void collect(final Object node, final List<UnaryOperator<String>> steps) {
    if (!(node instanceof Map<?, ?>)) {
      return;
    }
    Map<String, Object> map = Json.asObject(node);
    String type = Json.asString(map.get("type"));
    if (type == null) {
      return;
    }
    switch (type) {
      case "Sequence" -> {
        List<Object> nested = Json.asArray(map.get("normalizers"));
        if (nested != null) {
          nested.forEach(child -> collect(child, steps));
        }
      }
      case "NFC" -> steps.add(text -> Normalizer.normalize(text, NFC));
      case "NFD" -> steps.add(text -> Normalizer.normalize(text, NFD));
      case "NFKC" -> steps.add(text -> Normalizer.normalize(text, NFKC));
      case "NFKD" -> steps.add(text -> Normalizer.normalize(text, NFKD));
      case "Nmt" -> steps.add(TextNormalizer::nmt);
      case "Lowercase" -> steps.add(text -> text.toLowerCase(ROOT));
      case "Strip" -> steps.add(text -> strip(text, map));
      case "StripAccents" -> steps.add(TextNormalizer::stripAccents);
      case "Replace" -> steps.add(replace(map));
      case "BertNormalizer" -> steps.add(bert(map));
      case "Precompiled" -> steps.add(precompiled(map)::normalize);
      case "ByteLevel", "Identity" -> {
      }
      default -> {
      }
    }
  }

  private static String nmt(final String text) {
    StringBuilder out = new StringBuilder(text.length());
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      out.appendCodePoint(isNmtSpace(cp) ? ' ' : cp);
      i += Character.charCount(cp);
    }
    return out.toString();
  }

  private static boolean isNmtSpace(final int cp) {
    return cp == 0x000B || cp == 0x000C || cp == 0x00A0 || cp == 0x1680
      || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
      || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
  }

  private static String strip(final String text, final Map<String, Object> map) {
    boolean left = Json.asBoolean(map.get("strip_left"), true);
    boolean right = Json.asBoolean(map.get("strip_right"), true);
    if (left && right) {
      return text.strip();
    }
    if (left) {
      return text.stripLeading();
    }
    if (right) {
      return text.stripTrailing();
    }
    return text;
  }

  private static String stripAccents(final String text) {
    String decomposed = Normalizer.normalize(text, NFD);
    StringBuilder out = new StringBuilder(decomposed.length());
    for (int i = 0; i < decomposed.length(); ) {
      int cp = decomposed.codePointAt(i);
      if (Character.getType(cp) != Character.NON_SPACING_MARK) {
        out.appendCodePoint(cp);
      }
      i += Character.charCount(cp);
    }
    return out.toString();
  }

  private static UnaryOperator<String> replace(final Map<String, Object> map) {
    Object pattern = map.get("pattern");
    String content = Json.asString(map.get("content"));
    String replacement = content == null ? "" : content;
    if (pattern instanceof Map<?, ?> spec) {
      Object regex = spec.get("Regex");
      if (regex instanceof String regexText) {
        Pattern compiled = Pattern.compile(regexText);
        return text -> compiled.matcher(text).replaceAll(replacement);
      }
      Object literal = spec.get("String");
      if (literal instanceof String needle) {
        return text -> text.replace(needle, replacement);
      }
    }
    if (pattern instanceof String needle) {
      return text -> text.replace(needle, replacement);
    }
    return UnaryOperator.identity();
  }

  private static UnaryOperator<String> bert(final Map<String, Object> map) {
    boolean clean = Json.asBoolean(map.get("clean_text"), true);
    boolean chinese = Json.asBoolean(map.get("handle_chinese_chars"), true);
    boolean accents = Json.asBoolean(map.get("strip_accents"), false);
    boolean lower = Json.asBoolean(map.get("lowercase"), true);
    return text -> {
      String current = text;
      if (clean) {
        current = cleanBertText(current);
      }
      if (chinese) {
        current = isolateChinese(current);
      }
      if (accents) {
        current = stripAccents(current);
      }
      if (lower) {
        current = current.toLowerCase(ROOT);
      }
      return current;
    };
  }

  private static String cleanBertText(final String text) {
    StringBuilder out = new StringBuilder(text.length());
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      if (cp == 0 || cp == 0xFFFD || isBertControl(cp)) {
        i += Character.charCount(cp);
        continue;
      }
      out.appendCodePoint(Character.isWhitespace(cp) ? ' ' : cp);
      i += Character.charCount(cp);
    }
    return out.toString();
  }

  private static boolean isBertControl(final int cp) {
    return cp <= 0x1F || (cp >= 0x7F && cp <= 0x9F);
  }

  private static String isolateChinese(final String text) {
    StringBuilder out = new StringBuilder(text.length() + 8);
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      if (isChinese(cp)) {
        out.append(' ').appendCodePoint(cp).append(' ');
      } else {
        out.appendCodePoint(cp);
      }
      i += Character.charCount(cp);
    }
    return out.toString();
  }

  private static boolean isChinese(final int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)
      || (cp >= 0x3400 && cp <= 0x4DBF)
      || (cp >= 0x20000 && cp <= 0x2A6DF);
  }

  private static PrecompiledCharsMap precompiled(final Map<String, Object> map) {
    Object raw = map.get("precompiled_charsmap");
    if (raw == null) {
      throw new ModelLoadException("Precompiled normalizer is missing precompiled_charsmap");
    }
    return PrecompiledCharsMap.parse(charsmapBytes(raw));
  }

  static byte[] charsmapBytes(final Object raw) {
    if (raw instanceof List<?> list) {
      byte[] out = new byte[list.size()];
      for (int i = 0; i < list.size(); i++) {
        out[i] = (byte) Json.asInt(list.get(i), 0);
      }
      return out;
    }
    String text = Json.asString(raw);
    if (text == null || text.isEmpty()) {
      throw new ModelLoadException("precompiled charsmap is empty");
    }
    try {
      return Base64.getDecoder().decode(text);
    } catch (IllegalArgumentException ignored) {
      return text.getBytes(StandardCharsets.ISO_8859_1);
    }
  }

  String normalize(final String text) {
    String current = requireNonNull(text, "text");
    for (UnaryOperator<String> step : this.steps) {
      current = step.apply(current);
    }
    return current;
  }
}
