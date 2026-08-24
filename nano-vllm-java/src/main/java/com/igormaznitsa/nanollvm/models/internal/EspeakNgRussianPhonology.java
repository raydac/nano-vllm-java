package com.igormaznitsa.nanollvm.models.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Approximates espeak Russian phoneme programs after {@code ru_rules}: palatalization
 * before {@code и}/{@code е}, аканье/иканье, and syllable-count stress when unmarked.
 */
final class EspeakNgRussianPhonology {

  private static final Set<String> VOWELS = Set.of(
    "a", "A", "V", "e", "E", "E2", "E3", "E#", "e#", "i", "I", "I2", "I3", "I#",
    "o", "O", "8", "u", "U", "u#", "y", "0", "&", "@", "3", "02", "@-", "a#", "o#");
  private static final Set<String> PALATALIZING = Set.of(
    "i", "I", "I2", "I3", "I#", "e", "e#", "j", "8");
  private static final Set<String> PALATALIZABLE = Set.of(
    "b", "p", "v", "f", "g", "k", "d", "t", "z", "s", "x", "m", "n", "l", "r");
  private static final Set<String> HARD = Set.of("S", "Z", "ts");
  private static final Set<String> SKIP = Set.of("|", "%", "=", "#", "*", "_", "||");
  private static final Set<String> STOPS = Set.of("p", "b", "t", "d", "k", "g");
  private static final int[] GUESS = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11};
  private static final int[] GUESS_VOWEL_FINAL = {0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10};
  private static final int[] GUESS_STOP_FINAL = {0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10};

  private EspeakNgRussianPhonology() {
  }

  static String apply(final String phonemes) {
    return apply(phonemes, 0);
  }

  static String apply(final String phonemes, final int stressSyllable) {
    if (phonemes == null || phonemes.isEmpty()) {
      return phonemes == null ? "" : phonemes;
    }
    List<String> tokens = new ArrayList<>(EspeakNgPhonemes.tokens(phonemes));
    dropPalatalQuotes(tokens);
    markStress(tokens, stressSyllable);
    palatalize(tokens);
    reduceUnstressed(tokens);
    return String.join("", tokens);
  }

  private static void dropPalatalQuotes(final List<String> tokens) {
    for (int i = tokens.size() - 1; i >= 0; i--) {
      if (!"\"".equals(tokens.get(i))) {
        continue;
      }
      tokens.remove(i);
      int previous = previousSpoken(tokens, i);
      if (previous >= 0 && isPalatalizable(tokens.get(previous)) && !hasPalatal(tokens, previous)) {
        tokens.add(previous + 1, ";");
      }
    }
  }

  private static void markStress(final List<String> tokens, final int stressSyllable) {
    if (tokens.stream().anyMatch("'"::equals) || stressSyllable < 0) {
      return;
    }
    int vowel = stressSyllable == 0
      ? vowelAtGuessedSyllable(tokens)
      : nthVowel(tokens, stressSyllable);
    if (vowel >= 0) {
      tokens.add(vowel, "'");
    }
  }

  private static int vowelAtGuessedSyllable(final List<String> tokens) {
    int vowels = (int) tokens.stream().filter(EspeakNgRussianPhonology::isVowel).count();
    if (vowels <= 0) {
      return -1;
    }
    int capped = Math.min(vowels, GUESS.length - 1);
    String last = lastSpoken(tokens);
    int syllable = isVowel(last)
      ? GUESS_VOWEL_FINAL[capped]
      : STOPS.contains(last) ? GUESS_STOP_FINAL[capped] : GUESS[capped];
    return nthVowel(tokens, syllable <= 0 ? 1 : syllable);
  }

  private static String lastSpoken(final List<String> tokens) {
    for (int i = tokens.size() - 1; i >= 0; i--) {
      if (!isSkipped(tokens.get(i))) {
        return tokens.get(i);
      }
    }
    return "";
  }

  private static boolean isSkipped(final String token) {
    return SKIP.contains(token) || ";".equals(token) || "'".equals(token) || ",".equals(token);
  }

  private static int nthVowel(final List<String> tokens, final int syllable) {
    int seen = 0;
    int last = -1;
    for (int i = 0; i < tokens.size(); i++) {
      if (!isVowel(tokens.get(i))) {
        continue;
      }
      seen++;
      last = i;
      if (seen == syllable) {
        return i;
      }
    }
    return last;
  }

  private static void palatalize(final List<String> tokens) {
    for (int i = 0; i < tokens.size(); i++) {
      if (!isPalatalizable(tokens.get(i)) || hasPalatal(tokens, i)) {
        continue;
      }
      int vowel = nextSpoken(tokens, i + 1);
      while (vowel >= 0 && (";".equals(tokens.get(vowel)) || "'".equals(tokens.get(vowel))
        || ",".equals(tokens.get(vowel)))) {
        vowel = nextSpoken(tokens, vowel + 1);
      }
      if (vowel >= 0 && PALATALIZING.contains(tokens.get(vowel))) {
        tokens.add(i + 1, ";");
        i++;
      }
    }
  }

  private static void reduceUnstressed(final List<String> tokens) {
    boolean stressNextVowel = false;
    for (int i = 0; i < tokens.size(); i++) {
      String token = tokens.get(i);
      if ("'".equals(token)) {
        stressNextVowel = true;
        continue;
      }
      if (!isVowel(token)) {
        continue;
      }
      boolean stressed = stressNextVowel;
      stressNextVowel = false;
      if (stressed) {
        continue;
      }
      if ("o".equals(token) || "8".equals(token) || "a".equals(token) || "A".equals(token)) {
        tokens.set(i, "V");
        continue;
      }
      if ("e".equals(token) || "e#".equals(token)) {
        int previous = previousSpoken(tokens, i);
        while (previous >= 0 &&
          (";".equals(tokens.get(previous)) || "'".equals(tokens.get(previous)))) {
          previous = previousSpoken(tokens, previous);
        }
        tokens.set(i, previous >= 0 && HARD.contains(tokens.get(previous)) ? "E" : "i");
      }
    }
  }

  private static boolean hasPalatal(final List<String> tokens, final int consonant) {
    int next = consonant + 1;
    return next < tokens.size() && ";".equals(tokens.get(next));
  }

  private static int nextSpoken(final List<String> tokens, final int start) {
    for (int i = start; i < tokens.size(); i++) {
      if (!SKIP.contains(tokens.get(i))) {
        return i;
      }
    }
    return -1;
  }

  private static int previousSpoken(final List<String> tokens, final int start) {
    for (int i = start - 1; i >= 0; i--) {
      if (!SKIP.contains(tokens.get(i))) {
        return i;
      }
    }
    return -1;
  }

  private static boolean isVowel(final String token) {
    return VOWELS.contains(token);
  }

  private static boolean isPalatalizable(final String token) {
    return PALATALIZABLE.contains(token);
  }
}
