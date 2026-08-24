package com.igormaznitsa.nanollvm.models.internal;

import static java.text.Normalizer.Form.NFD;
import static java.util.Comparator.comparingInt;
import static java.util.Map.entry;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Kirschenbaum / espeak mnemonics from {@code *_list} / {@code *_rules} to IPA.
 */
final class EspeakNgPhonemes {

  private static final List<Map.Entry<String, String>> MNEMONICS = List.of(
      entry("aI", "aɪ"),
      entry("aU", "aʊ"),
      entry("eI", "eɪ"),
      entry("oU", "oʊ"),
      entry("OI", "ɔɪ"),
      entry("I@", "ɪə"),
      entry("e@", "eə"),
      entry("U@", "ʊə"),
      entry("@-", "ɨ"),
      entry("3:", "ɜː"),
      entry("A:", "ɑː"),
      entry("O:", "ɔː"),
      entry("i:", "iː"),
      entry("u:", "uː"),
      entry("o:", "oː"),
      entry("e:", "eː"),
      entry("E:", "ɛː"),
      entry("y:", "yː"),
      entry("02", "ʌ"),
      entry("I2", "ɪ"),
      entry("I3", "ɪ"),
      entry("I#", "i"),
      entry("E2", "ɛ"),
      entry("E3", "ɛ"),
      entry("E#", "ɛ"),
      entry("a#", "a"),
      entry("o#", "o"),
      entry("e#", "e"),
      entry("u#", "u"),
      entry("tS;", "tɕ"),
      entry("dZ;", "dʑ"),
      entry("S;", "ɕ"),
      entry("Z;", "ʑ"),
      entry("tS", "tʃ"),
      entry("dZ", "dʒ"),
      entry("ts", "ts"),
      entry("dz", "dz"),
      entry("@L", "əl"),
      entry("l/", "ɫ"),
      entry("n^", "ɲ"),
      entry("A", "ɑ"),
      entry("O", "ɔ"),
      entry("E", "ɛ"),
      entry("I", "ɪ"),
      entry("U", "ʊ"),
      entry("V", "ʌ"),
      entry("&", "æ"),
      entry("@", "ə"),
      entry("3", "ɜ"),
      entry("0", "ɒ"),
      entry("8", "o"),
      entry("D", "ð"),
      entry("T", "θ"),
      entry("S", "ʃ"),
      entry("Z", "ʒ"),
      entry("N", "ŋ"),
      entry("R", "r"),
      entry("L", "l"),
      entry("Q", "ʔ"),
      entry("?", "ʔ"),
      entry(";", "ʲ"),
      entry(":", "ː"),
      entry("'", "ˈ"),
      entry(",", "ˌ"),
      entry("Y", "y"),
      entry("W", "w")
    ).stream()
    .sorted(comparingInt((Map.Entry<String, String> e) -> e.getKey().length()).reversed())
    .toList();

  private EspeakNgPhonemes() {
  }

  static String toIpa(final String kirschenbaum) {
    if (kirschenbaum == null || kirschenbaum.isEmpty()) {
      return "";
    }
    StringBuilder ipa = new StringBuilder(kirschenbaum.length() * 2);
    int i = 0;
    while (i < kirschenbaum.length()) {
      if (kirschenbaum.startsWith("||", i)) {
        ipa.append(' ');
        i += 2;
        continue;
      }
      String mnemonic = longestMnemonic(kirschenbaum, i);
      if (mnemonic != null) {
        ipa.append(valueOf(mnemonic));
        i += mnemonic.length();
        continue;
      }
      int cp = kirschenbaum.codePointAt(i);
      int width = Character.charCount(cp);
      if (cp == '|' || cp == '%' || cp == '=' || cp == '#' || cp == '*' || cp == '_') {
        i += width;
        continue;
      }
      ipa.appendCodePoint(cp);
      i += width;
    }
    return Normalizer.normalize(ipa, NFD);
  }

  static List<String> tokens(final String kirschenbaum) {
    List<String> tokens = new ArrayList<>();
    int i = 0;
    while (i < kirschenbaum.length()) {
      if (kirschenbaum.startsWith("||", i)) {
        tokens.add("||");
        i += 2;
        continue;
      }
      String mnemonic = longestMnemonic(kirschenbaum, i);
      if (mnemonic != null) {
        tokens.add(mnemonic);
        i += mnemonic.length();
        continue;
      }
      int cp = kirschenbaum.codePointAt(i);
      tokens.add(new String(Character.toChars(cp)));
      i += Character.charCount(cp);
    }
    return List.copyOf(tokens);
  }

  static String longestMnemonic(final String text, final int start) {
    return MNEMONICS.stream()
      .map(Map.Entry::getKey)
      .filter(key -> text.startsWith(key, start))
      .findFirst()
      .orElse(null);
  }

  private static String valueOf(final String mnemonic) {
    return MNEMONICS.stream()
      .filter(entry -> entry.getKey().equals(mnemonic))
      .map(Map.Entry::getValue)
      .findFirst()
      .orElse("");
  }
}
