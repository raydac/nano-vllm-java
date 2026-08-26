package com.igormaznitsa.nanollvm.models.internal;

import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Maps British-leaning espeak IPA toward Piper en-us phones (rhotic {@code ɹ}/{@code ɚ},
 * LOT as {@code ɑː}, stress on content words) without running phoneme programs.
 */
final class EspeakNgEnglishPhonology {

  private static final Pattern WORD_BREAK = Pattern.compile(" +");
  private static final Pattern DIPHTHONG_SCHWA =
    Pattern.compile("(aɪ|aʊ|eɪ|oʊ|ɔɪ)ə(?=[bdfghjklmnpstvwzŋʃʒθðɹɫ])");
  private static final Set<String> SCHWA = Set.of("ə", "ɚ", "ɨ");
  private static final List<String> VOWELS = List.of(
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "ɑː", "ɔː", "iː", "uː", "eː", "oː", "ɜː", "ɝː",
    "ɑ", "ɒ", "æ", "ɛ", "e", "i", "ɪ", "o", "ɔ", "u", "ʊ", "ʌ", "ɝ", "ɚ", "ə", "ɜ", "ɵ");

  private EspeakNgEnglishPhonology() {
  }

  static String apply(final String ipa) {
    if (ipa == null || ipa.isEmpty()) {
      return ipa == null ? "" : ipa;
    }
    String remapped = DIPHTHONG_SCHWA.matcher(ipa)
      .replaceAll("$1")
      .replace('r', 'ɹ')
      .replace("ɒ", "ɑː")
      .replace("ɜː", "ɝː")
      .replace('ɜ', 'ɚ');
    StringBuilder out = new StringBuilder(remapped.length() + 4);
    for (String word : WORD_BREAK.split(remapped, -1)) {
      if (word.isEmpty()) {
        continue;
      }
      if (!out.isEmpty()) {
        out.append(' ');
      }
      out.append(stressWord(word));
    }
    return out.toString();
  }

  private static String stressWord(final String word) {
    if (word.indexOf('ˈ') >= 0 || word.indexOf('ˌ') >= 0) {
      return word;
    }
    int vowel = firstFullVowel(word);
    return vowel < 0 ? word : word.substring(0, vowel) + 'ˈ' + word.substring(vowel);
  }

  private static int firstFullVowel(final String word) {
    int i = 0;
    while (i < word.length()) {
      String vowel = longestVowel(word, i);
      if (vowel != null) {
        if (!SCHWA.contains(vowel)) {
          return i;
        }
        i += vowel.length();
        continue;
      }
      i += Character.charCount(word.codePointAt(i));
    }
    return -1;
  }

  private static String longestVowel(final String word, final int start) {
    return VOWELS.stream()
      .filter(vowel -> word.startsWith(vowel, start))
      .findFirst()
      .orElse(null);
  }
}
