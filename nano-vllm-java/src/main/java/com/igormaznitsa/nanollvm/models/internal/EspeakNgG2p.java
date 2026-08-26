package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.empty;

import java.lang.Character.UnicodeScript;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * Text → IPA phoneme ids using Piper's {@code phoneme_id_map}. When
 * {@code espeak-ng-data/dictsource} has {@code *_list} / {@code *_rules} for the voice,
 * those files are the G2P. If {@code dictsource} is missing but compiled {@code phontab}
 * and {@code *_dict} are present, those are used for listed words. Otherwise a Java
 * letter-to-sound front-end is used (Cyrillic Russian orthoepy; Latin a small English
 * lexicon plus letter/digraph rules). Digits use {@code *_list} number fragments when
 * present. Suffix/prefix {@code S}/{@code P} rules retranslate the stem.
 * {@code espeak-ng-data} from {@link com.igormaznitsa.nanollvm.models.LlmOptionalData#ESPEAK_DATA}
 * is optional: a missing or incomplete directory is ignored.
 *
 * @since 1.3.0
 */
final class EspeakNgG2p {

  private static final Pattern ENGLISH_WORD_BREAK = Pattern.compile("[\\s\\-–—]+");
  private static final String PALATAL = "ʲ";
  private static final String VOICED_PAIRED = "бвгджз";
  private static final String VOICELESS_PAIRED = "пфктшс";
  private static final String VOICELESS_UNPAIRED = "хцчщ";
  private static final Set<String> KEEP_G_IN_OGO = Set.of(
    "много", "немного", "намного", "премного",
    "строго", "нестрого", "настрого",
    "дорого", "недорого", "задорого",
    "полого", "убого", "лего", "лого", "ого");
  private static final Map<String, String> ENGLISH_WORDS = Map.ofEntries(
    Map.entry("a", "ə"),
    Map.entry("and", "ænd"),
    Map.entry("english", "ˈɪŋɡlɪʃ"),
    Map.entry("good", "ɡʊd"),
    Map.entry("hello", "həlˈoʊ"),
    Map.entry("is", "ɪz"),
    Map.entry("morning", "mˈɔrnɪŋ"),
    Map.entry("piper", "pˈaɪpɚ"),
    Map.entry("test", "tɛst"),
    Map.entry("the", "ðə"),
    Map.entry("this", "ðɪs"),
    Map.entry("world", "wˈɝld"),
    Map.entry("you", "ju"));

  private final Map<String, List<Integer>> phonemeIdMap;
  private final List<String> keysLongestFirst;
  private final Path dataDir;
  private final EspeakNgDictSource dictionary;
  private final boolean englishUs;

  EspeakNgG2p(final Map<String, List<Integer>> phonemeIdMap, final Path dataDir) {
    this(phonemeIdMap, dataDir, "");
  }

  EspeakNgG2p(
    final Map<String, List<Integer>> phonemeIdMap,
    final Path dataDir,
    final String espeakVoice
  ) {
    this.phonemeIdMap = Map.copyOf(requireNonNull(phonemeIdMap, "phonemeIdMap"));
    this.dataDir = requireNonNull(dataDir, "dataDir").toAbsolutePath().normalize();
    this.keysLongestFirst = this.phonemeIdMap.keySet().stream()
      .sorted(Comparator.comparingInt(String::length).reversed().thenComparing(key -> key))
      .toList();
    String voice = requireNonNull(espeakVoice, "espeakVoice");
    this.dictionary = EspeakNgDictSource.load(this.dataDir, voice);
    this.englishUs = isEnglishUs(voice);
  }

  private static boolean isEnglishUs(final String voice) {
    String folded = voice.strip().toLowerCase(ROOT);
    return folded.equals("en") || folded.startsWith("en-") || folded.startsWith("en_");
  }

  private static boolean isCyrillic(final int cp) {
    return UnicodeScript.of(cp) == UnicodeScript.CYRILLIC;
  }

  private static boolean isLatin(final int cp) {
    return UnicodeScript.of(cp) == UnicodeScript.LATIN;
  }

  private static boolean isEnglishVowel(final int cp) {
    return "aeiouy".indexOf(cp) >= 0;
  }

  private static boolean isVoiced(final char ch) {
    return VOICED_PAIRED.indexOf(ch) >= 0;
  }

  private static boolean isPairedObstruent(final char ch) {
    return isVoiced(ch) || VOICELESS_PAIRED.indexOf(ch) >= 0;
  }

  private static boolean isObstruent(final char ch) {
    return isPairedObstruent(ch) || VOICELESS_UNPAIRED.indexOf(ch) >= 0;
  }

  private static char destress(final char ch) {
    int index = VOICED_PAIRED.indexOf(ch);
    return index < 0 ? ch : VOICELESS_PAIRED.charAt(index);
  }

  private static char voice(final char ch) {
    int index = VOICELESS_PAIRED.indexOf(ch);
    return index < 0 ? ch : VOICED_PAIRED.charAt(index);
  }

  /**
   * Maps {@code text} to Piper phoneme ids ({@code ^ _ … _ $} framing).
   *
   * @param text utterance; must not be {@code null}
   * @return id sequence; never empty after framing
   */
  List<Integer> phonemeIds(final CharSequence text) {
    String ipa = this.toIpa(requireNonNull(text, "text").toString());
    List<Integer> ids = new ArrayList<>();
    this.appendSymbol(ids, "^");
    this.appendSymbol(ids, "_");
    int i = 0;
    while (i < ipa.length()) {
      String match = this.longestKey(ipa, i);
      if (match == null) {
        match = this.ipaVelarIfUnmapped(ipa, i);
      }
      if (match == null) {
        i += Character.charCount(ipa.codePointAt(i));
        continue;
      }
      this.appendSymbol(ids, match);
      this.appendSymbol(ids, "_");
      i += match.length();
    }
    this.appendSymbol(ids, "$");
    if (ids.isEmpty()) {
      throw new IllegalArgumentException("no phoneme ids produced for text");
    }
    return List.copyOf(ids);
  }

  boolean hasEspeakData() {
    return Files.isDirectory(this.dataDir) && Files.isDirectory(this.dataDir.resolve("lang"));
  }

  private void appendSymbol(final List<Integer> ids, final String symbol) {
    List<Integer> mapped = this.phonemeIdMap.get(symbol);
    if (mapped != null) {
      ids.addAll(mapped);
    }
  }

  private String longestKey(final String ipa, final int start) {
    return this.keysLongestFirst.stream()
      .filter(key -> ipa.startsWith(key, start))
      .findFirst()
      .orElse(null);
  }

  private String ipaVelarIfUnmapped(final String ipa, final int start) {
    return ipa.charAt(start) == 'g' && this.phonemeIdMap.containsKey("ɡ") ? "ɡ" : null;
  }

  private String toIpa(final String text) {
    String folded = text.strip().toLowerCase(ROOT);
    if (folded.isEmpty()) {
      return "";
    }
    StringBuilder ipa = new StringBuilder(folded.length() * 2);
    int[] cps = folded.codePoints().toArray();
    int i = 0;
    while (i < cps.length) {
      int cp = cps[i];
      if (Character.isWhitespace(cp) || cp == '-' || cp == '–' || cp == '—') {
        ipa.append(' ');
        i++;
        continue;
      }
      if (this.isPunctuation(cp)) {
        i++;
        continue;
      }
      if (Character.isDigit(cp)) {
        int end = this.spanDigits(cps, i);
        ipa.append(this.digitsToIpa(new String(cps, i, end - i)));
        i = end;
        continue;
      }
      if (isCyrillic(cp)) {
        int end = this.spanSameScript(cps, i, true);
        ipa.append(this.spanToIpa(new String(cps, i, end - i), true));
        i = end;
        continue;
      }
      if (isLatin(cp)) {
        int end = this.spanSameScript(cps, i, false);
        ipa.append(this.spanToIpa(new String(cps, i, end - i), false));
        i = end;
        continue;
      }
      i++;
    }
    return ipa.toString();
  }

  private String spanToIpa(final String span, final boolean cyrillic) {
    if (cyrillic) {
      return this.dictionary.isLoaded()
        ? this.wordsToIpa(span, this::cyrillicWordToIpa)
        : this.russianSpanToIpa(span);
    }
    return this.wordsToIpa(span, this::latinWordToIpa);
  }

  private String wordsToIpa(final String span, final Function<String, String> wordToIpa) {
    StringBuilder ipa = new StringBuilder(span.length() * 2);
    for (String word : ENGLISH_WORD_BREAK.split(span, -1)) {
      if (word.isEmpty()) {
        continue;
      }
      String mapped = wordToIpa.apply(word);
      if (mapped.isEmpty()) {
        continue;
      }
      if (!ipa.isEmpty() && ipa.charAt(ipa.length() - 1) != ' ') {
        ipa.append(' ');
      }
      ipa.append(mapped);
    }
    return ipa.toString();
  }

  private String cyrillicWordToIpa(final String word) {
    return this.dictionaryWord(word).orElseGet(() -> this.russianSpanToIpa(word));
  }

  private String latinWordToIpa(final String word) {
    String letters = word.replace("'", "");
    return this.dictionaryWord(letters).orElseGet(() -> this.englishWordToIpa(letters));
  }

  private Optional<String> dictionaryWord(final String word) {
    if (!this.dictionary.isLoaded()) {
      return empty();
    }
    return this.dictionary.phonemesForWord(word.toLowerCase(ROOT)).map(this::kirschenbaumToIpa);
  }

  private String kirschenbaumToIpa(final String kirschenbaum) {
    String ipa = EspeakNgPhonemes.toIpa(kirschenbaum);
    return this.englishUs ? EspeakNgEnglishPhonology.apply(ipa) : ipa;
  }

  private String englishWordToIpa(final String word) {
    String known = ENGLISH_WORDS.get(word);
    String ipa = known != null ? known : this.englishLetters(word);
    return this.englishUs ? EspeakNgEnglishPhonology.apply(ipa) : ipa;
  }

  private String russianSpanToIpa(final String span) {
    String letters = this.assimilateObstruents(this.rewriteOrthography(span));
    StringBuilder ipa = new StringBuilder(letters.length() * 2);
    int[] cps = letters.codePoints().toArray();
    for (int i = 0; i < cps.length; i++) {
      int cp = cps[i];
      if (Character.isWhitespace(cp) || cp == '-' || cp == '–' || cp == '—') {
        ipa.append(' ');
        continue;
      }
      if (this.isPunctuation(cp)) {
        continue;
      }
      ipa.append(this.russianGrapheme(cps, i));
    }
    return ipa.toString();
  }

  private String englishLetters(final String word) {
    StringBuilder ipa = new StringBuilder(word.length() * 2);
    int[] cps = word.codePoints().toArray();
    int i = 0;
    while (i < cps.length) {
      int consumed = this.appendEnglishGrapheme(ipa, cps, i);
      i += Math.max(consumed, 1);
    }
    return ipa.toString();
  }

  private int appendEnglishGrapheme(final StringBuilder ipa, final int[] cps, final int index) {
    String rest = new String(cps, index, cps.length - index);
    if (rest.startsWith("tch")) {
      ipa.append("tʃ");
      return 3;
    }
    if (rest.startsWith("sch")) {
      ipa.append("sk");
      return 3;
    }
    if (rest.startsWith("the") && cps.length - index == 3) {
      ipa.append("ðə");
      return 3;
    }
    if (rest.startsWith("th")) {
      ipa.append(index == 0 ? "θ" : "ð");
      return 2;
    }
    if (rest.startsWith("sh")) {
      ipa.append("ʃ");
      return 2;
    }
    if (rest.startsWith("ch")) {
      ipa.append("tʃ");
      return 2;
    }
    if (rest.startsWith("ng")) {
      ipa.append("ŋ");
      return 2;
    }
    if (rest.startsWith("ck") || rest.startsWith("qu")) {
      ipa.append(rest.startsWith("qu") ? "kw" : "k");
      return 2;
    }
    if (rest.startsWith("ph") || (rest.startsWith("gh") && index > 0)) {
      ipa.append("f");
      return 2;
    }
    if ((rest.startsWith("kn") || rest.startsWith("wr") || rest.startsWith("wh")) && index == 0) {
      ipa.append(rest.startsWith("wh") ? "w" : rest.startsWith("kn") ? "n" : "r");
      return 2;
    }
    if (rest.startsWith("ee") || rest.startsWith("ea")) {
      ipa.append("i");
      return 2;
    }
    if (rest.startsWith("oo")) {
      ipa.append("u");
      return 2;
    }
    if (rest.startsWith("ou") || rest.startsWith("ow")) {
      ipa.append("aʊ");
      return 2;
    }
    if (rest.startsWith("ay") || rest.startsWith("ai") || rest.startsWith("ey")) {
      ipa.append("eɪ");
      return 2;
    }
    if (rest.startsWith("oy") || rest.startsWith("oi")) {
      ipa.append("ɔɪ");
      return 2;
    }
    if (rest.startsWith("aw") || rest.startsWith("au")) {
      ipa.append("ɔ");
      return 2;
    }
    if (rest.startsWith("oa")) {
      ipa.append("oʊ");
      return 2;
    }
    if (rest.startsWith("er") || rest.startsWith("ir") || rest.startsWith("ur")) {
      ipa.append("ɚ");
      return 2;
    }
    if (rest.startsWith("ar")) {
      ipa.append("ɑr");
      return 2;
    }
    if (rest.startsWith("or")) {
      ipa.append("ɔr");
      return 2;
    }
    if (rest.startsWith("ew")) {
      ipa.append("ju");
      return 2;
    }
    return this.appendEnglishLetter(ipa, cps, index);
  }

  private int appendEnglishLetter(final StringBuilder ipa, final int[] cps, final int index) {
    int cp = cps[index];
    int next = index + 1 < cps.length ? cps[index + 1] : 0;
    boolean last = index == cps.length - 1;
    boolean silentE = last && cp == 'e' && cps.length > 2 && !isEnglishVowel(cps[index - 1]);
    if (silentE) {
      return 1;
    }
    switch (cp) {
      case 'a' -> ipa.append(last ? "ə" : "æ");
      case 'e' -> ipa.append("ɛ");
      case 'i' -> ipa.append("ɪ");
      case 'o' -> ipa.append(last ? "oʊ" : "ɑ");
      case 'u' -> ipa.append("ʌ");
      case 'y' -> ipa.append(index == 0 ? "j" : last ? "aɪ" : "ɪ");
      case 'c' -> ipa.append((next == 'e' || next == 'i' || next == 'y') ? "s" : "k");
      case 'g' -> ipa.append((next == 'e' || next == 'i' || next == 'y') ? "dʒ" : "ɡ");
      case 'j' -> ipa.append("dʒ");
      case 'q' -> ipa.append("k");
      case 'x' -> ipa.append("ks");
      default -> {
        if (cp >= 'a' && cp <= 'z') {
          ipa.appendCodePoint(cp == 'g' ? 'ɡ' : cp);
        }
      }
    }
    return 1;
  }

  private String digitsToIpa(final String digits) {
    return this.dictionaryWord(digits).orElseGet(() -> this.spellDigits(digits));
  }

  private String spellDigits(final String digits) {
    StringBuilder ipa = new StringBuilder();
    for (int i = 0; i < digits.length(); i++) {
      char ch = digits.charAt(i);
      if (ch == '.' || ch == ',') {
        if (!ipa.isEmpty()) {
          ipa.append(' ');
        }
        continue;
      }
      String mapped = this.dictionaryWord(String.valueOf(ch)).orElse("");
      if (mapped.isEmpty()) {
        continue;
      }
      if (!ipa.isEmpty()) {
        ipa.append(' ');
      }
      ipa.append(mapped);
    }
    return ipa.toString();
  }

  private int spanDigits(final int[] cps, final int start) {
    int i = start;
    boolean seenSep = false;
    while (i < cps.length) {
      int cp = cps[i];
      if (Character.isDigit(cp)) {
        i++;
        continue;
      }
      if (!seenSep && (cp == '.' || cp == ',') && i + 1 < cps.length &&
        Character.isDigit(cps[i + 1])) {
        seenSep = true;
        i++;
        continue;
      }
      break;
    }
    return i;
  }

  private int spanSameScript(final int[] cps, final int start, final boolean cyrillic) {
    int i = start;
    int lastLetter = start;
    while (i < cps.length) {
      int cp = cps[i];
      if (cyrillic ? isCyrillic(cp) : isLatin(cp)) {
        lastLetter = i;
        i++;
        continue;
      }
      if (!cyrillic && cp == '\'') {
        i++;
        continue;
      }
      if (Character.isWhitespace(cp) || cp == '-' || cp == '–' || cp == '—') {
        int j = i + 1;
        while (j < cps.length && (Character.isWhitespace(cps[j]) || cps[j] == '-' || cps[j] == '–'
          || cps[j] == '—')) {
          j++;
        }
        if (j < cps.length && (cyrillic ? isCyrillic(cps[j]) : isLatin(cps[j]))) {
          i = j;
          continue;
        }
        break;
      }
      break;
    }
    return lastLetter + 1;
  }

  private boolean isPunctuation(final int cp) {
    return cp == '.' || cp == ',' || cp == '!' || cp == '?' || cp == ';' || cp == ':' || cp == '…';
  }

  private String russianGrapheme(final int[] cps, final int index) {
    int cp = cps[index];
    boolean palatalize = this.softensPrevious(this.nextCp(cps, index));
    return switch (cp) {
      case 'а' -> "a";
      case 'б' -> palatalize ? "b" + PALATAL : "b";
      case 'в' -> palatalize ? "v" + PALATAL : "v";
      case 'г' -> palatalize ? "ɡ" + PALATAL : "ɡ";
      case 'д' -> palatalize ? "d" + PALATAL : "d";
      case 'е' -> this.iotated("e", index, cps);
      case 'ё' -> this.iotated("o", index, cps);
      case 'ж' -> "ʒ";
      case 'з' -> palatalize ? "z" + PALATAL : "z";
      case 'и' -> "i";
      case 'й' -> "j";
      case 'к' -> palatalize ? "k" + PALATAL : "k";
      case 'л' -> palatalize ? "l" + PALATAL : "l";
      case 'м' -> palatalize ? "m" + PALATAL : "m";
      case 'н' -> palatalize ? "n" + PALATAL : "n";
      case 'о' -> "o";
      case 'п' -> palatalize ? "p" + PALATAL : "p";
      case 'р' -> palatalize ? "r" + PALATAL : "r";
      case 'с' -> palatalize ? "s" + PALATAL : "s";
      case 'т' -> palatalize ? "t" + PALATAL : "t";
      case 'у' -> "u";
      case 'ф' -> palatalize ? "f" + PALATAL : "f";
      case 'х' -> palatalize ? "x" + PALATAL : "x";
      case 'ц' -> "ts";
      case 'ч' -> "tɕ";
      case 'ш' -> "ʃ";
      case 'щ' -> "ɕ";
      case 'ъ' -> "";
      case 'ы' -> "y";
      case 'ь' -> "";
      case 'э' -> "ɛ";
      case 'ю' -> this.iotated("u", index, cps);
      case 'я' -> this.iotated("a", index, cps);
      default -> Character.isLetter(cp) ? String.valueOf(Character.toChars(cp)) : "";
    };
  }

  private String rewriteOrthography(final String text) {
    StringBuilder out = new StringBuilder(text.length());
    int i = 0;
    while (i < text.length()) {
      char ch = text.charAt(i);
      if (!Character.isLetter(ch)) {
        out.append(ch);
        i++;
        continue;
      }
      int end = i + 1;
      while (end < text.length() && Character.isLetter(text.charAt(end))) {
        end++;
      }
      out.append(this.rewriteWord(text.substring(i, end)));
      i = end;
    }
    return out.toString();
  }

  private String rewriteWord(final String word) {
    if (word.equals("сегодня")) {
      return "севодня";
    }
    if (word.equals("бог")) {
      return "бох";
    }
    String rewritten = word.replace("жи", "жы").replace("ши", "шы").replace("ци", "цы");
    rewritten = this.rewriteOgoEgo(rewritten);
    rewritten = rewritten.replace("что", "што");
    if (rewritten.endsWith("ться")) {
      rewritten = rewritten.substring(0, rewritten.length() - 4) + "ца";
    } else if (rewritten.endsWith("тся")) {
      rewritten = rewritten.substring(0, rewritten.length() - 3) + "ца";
    }
    return rewritten.replace("сч", "щ").replace("зч", "щ").replace("гк", "хк").replace("гч", "хч");
  }

  private String rewriteOgoEgo(final String word) {
    if (KEEP_G_IN_OGO.contains(word) || word.length() < 3 || !word.endsWith("го")) {
      return word;
    }
    char beforeG = word.charAt(word.length() - 3);
    if (beforeG != 'е' && beforeG != 'о') {
      return word;
    }
    return word.substring(0, word.length() - 2) + "во";
  }

  private String assimilateObstruents(final String text) {
    char[] chars = text.toCharArray();
    this.destressWordFinals(chars);
    this.assimilateClusters(chars);
    return new String(chars);
  }

  private void destressWordFinals(final char[] chars) {
    int i = 0;
    while (i < chars.length) {
      if (!Character.isLetter(chars[i])) {
        i++;
        continue;
      }
      int end = i + 1;
      while (end < chars.length && Character.isLetter(chars[end])) {
        end++;
      }
      int last = end - 1;
      while (last >= i && (chars[last] == 'ь' || chars[last] == 'ъ')) {
        last--;
      }
      if (last >= i) {
        chars[last] = destress(chars[last]);
      }
      i = end;
    }
  }

  private void assimilateClusters(final char[] chars) {
    int i = 0;
    while (i < chars.length) {
      if (!isObstruent(chars[i])) {
        i++;
        continue;
      }
      int lastObstruent = i;
      int j = i + 1;
      while (j < chars.length) {
        char ch = chars[j];
        if (isObstruent(ch)) {
          lastObstruent = j;
          j++;
        } else if (ch == ' ' || ch == 'ь' || ch == 'ъ') {
          j++;
        } else {
          break;
        }
      }
      boolean voiced = isVoiced(chars[lastObstruent]);
      for (int k = i; k <= lastObstruent; k++) {
        if (isPairedObstruent(chars[k])) {
          chars[k] = voiced ? voice(chars[k]) : destress(chars[k]);
        }
      }
      i = j;
    }
  }

  private String iotated(final String vowel, final int index, final int[] cps) {
    int prev = index == 0 ? 0 : cps[index - 1];
    boolean jot = index == 0 || Character.isWhitespace(prev) || prev == 'ъ' || prev == 'ь'
      || this.isVowel(prev);
    return jot ? "j" + vowel : vowel;
  }

  private boolean softensPrevious(final int next) {
    return next == 'е' || next == 'ё' || next == 'и' || next == 'ю' || next == 'я' || next == 'ь';
  }

  private boolean isVowel(final int cp) {
    return "аеёиоуыэюяaeiouy".indexOf(cp) >= 0;
  }

  private int nextCp(final int[] cps, final int index) {
    return index + 1 < cps.length ? cps[index + 1] : 0;
  }
}
