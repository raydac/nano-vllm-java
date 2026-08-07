package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Load-time preparsing: Unicode normalize, separate model text from search text,
 * inject source-stem tokens, precompute term frequencies for BM25.
 */
public final class PassagePreparser {

  private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final Pattern STEM_SPLIT = Pattern.compile("[^\\p{L}\\p{N}]+");

  private PassagePreparser() {
  }

  public static List<PreparedPassage> prepare(final List<TextChunk> chunks) {
    requireNonNull(chunks, "chunks");
    List<PreparedPassage> prepared = new ArrayList<>(chunks.size());
    for (TextChunk chunk : chunks) {
      if (!chunk.isBlank()) {
        prepared.add(prepareOne(chunk));
      }
    }
    if (prepared.isEmpty()) {
      throw new IllegalArgumentException("no non-blank passages to prepare");
    }
    return List.copyOf(prepared);
  }

  public static PreparedPassage prepareOne(final TextChunk chunk) {
    requireNonNull(chunk, "chunk");
    String modelText = normalizeModelText(chunk.text());
    TextChunk normalized = new TextChunk(chunk.id(), chunk.source(), modelText);
    String searchText = buildSearchText(modelText, chunk.source());
    Map<String, Integer> tf = termFrequencies(searchText);
    int tokens = tf.values().stream().mapToInt(Integer::intValue).sum();
    return new PreparedPassage(normalized, searchText, tf, tokens);
  }

  static String normalizeModelText(final String raw) {
    if (raw == null || raw.isBlank()) {
      return "";
    }
    String text = Normalizer.normalize(raw, Normalizer.Form.NFC);
    return text.replace('\u00A0', ' ').replaceAll("\\s+", " ").strip();
  }

  static String buildSearchText(final String modelText, final String source) {
    StringBuilder search = new StringBuilder(modelText);
    for (String stemToken : sourceStemTokens(source)) {
      search.append(' ').append(stemToken);
    }
    return search.toString();
  }

  static List<String> sourceStemTokens(final String source) {
    if (source == null || source.isBlank()) {
      return List.of();
    }
    String name;
    try {
      Path fileName = Path.of(source).getFileName();
      name = fileName == null ? source : fileName.toString();
    } catch (RuntimeException e) {
      name = source;
    }
    int dot = name.lastIndexOf('.');
    if (dot > 0) {
      name = name.substring(0, dot);
    }
    name = name.toLowerCase(Locale.ROOT);
    List<String> tokens = new ArrayList<>();
    for (String part : STEM_SPLIT.split(name)) {
      if (part.length() > 1) {
        tokens.add(part);
      }
    }
    if (tokens.isEmpty() && name.length() > 1) {
      tokens.add(name);
    }
    return List.copyOf(tokens);
  }

  static List<String> tokenize(final String text) {
    List<String> tokens = new ArrayList<>();
    var matcher = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
    while (matcher.find()) {
      String token = matcher.group();
      if (token.length() <= 1) {
        continue;
      }
      tokens.add(token);
      addInflectionKeys(token, tokens);
    }
    return tokens;
  }

  /**
   * Extra keys so Russian case endings still hit ({@code колобок} ↔ {@code колобке}).
   * Applied only to Cyrillic tokens so English query-length heuristics stay stable.
   */
  private static void addInflectionKeys(final String token, final List<String> tokens) {
    if (!isCyrillicToken(token)) {
      return;
    }
    if (token.length() >= 4) {
      tokens.add(token.substring(0, token.length() - 1));
    }
    if (token.length() >= 5) {
      String prefix = token.substring(0, 5);
      if (!prefix.equals(token)) {
        tokens.add(prefix);
      }
    }
  }

  private static boolean isCyrillicToken(final String token) {
    return token.chars().anyMatch(c -> c >= 0x0400 && c <= 0x04FF);
  }

  static Map<String, Integer> termFrequencies(final String text) {
    Map<String, Integer> tf = new HashMap<>();
    for (String token : tokenize(text)) {
      tf.merge(token, 1, Integer::sum);
    }
    return Map.copyOf(tf);
  }
}
