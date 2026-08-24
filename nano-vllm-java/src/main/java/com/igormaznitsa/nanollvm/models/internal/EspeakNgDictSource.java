package com.igormaznitsa.nanollvm.models.internal;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparingInt;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toUnmodifiableSet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Reads espeak-ng {@code dictsource} ({@code *_list} / {@code *_rules}) or compiled
 * {@code *_dict} / {@code phontab} when the source files are absent.
 */
final class EspeakNgDictSource {

  private static final Pattern WORD_BREAK = Pattern.compile("[\\s\\-–—]+");
  private static final Pattern WHITESPACE = Pattern.compile("\\s+");
  private static final Pattern NUMBER_KEY = Pattern.compile(
    "_(?:\\d{1,2}|\\dx|\\dc0?|0c|0m\\d{1,2}|0and|dpt2?)", Pattern.CASE_INSENSITIVE);
  private static final Pattern DIGITS = Pattern.compile("[0-9]+(?:[.,][0-9]+)?");
  private static final String LATIN_VOWELS = "aeiou";
  private static final String CYRILLIC_VOWELS = "аеёиоуыэюяәөұүі";
  private static final Map<Integer, Set<Integer>> RUSSIAN_LETTER_GROUPS = Map.of(
    (int) 'B', codePoints("ьйчщ"),
    (int) 'F', codePoints("бвгдзйклмнпрстфхчщь"),
    (int) 'G', codePoints("бвгджз"),
    (int) 'H', codePoints("ъжцш"),
    (int) 'Y', codePoints("ьюяёеи"));
  private static final int MAX_AFFIX_DEPTH = 8;
  private static final int FLAG_E = 1;
  private static final int FLAG_I = 2;
  private static final int FLAG_D = 4;
  private static final int FLAG_Q = 8;
  private static final int FLAG_M = 16;
  private static final Set<String> ADD_E_EXCEPTIONS = Set.of("ion");
  private static final Set<String> ADD_E_ADDITIONS = Set.of(
    "c", "rs", "ir", "ur", "ath", "ns", "u", "spong", "rang", "larg");
  private static final Set<String> CONTEXT_FLAGS = Set.of(
    "$atend", "$atstart", "$verb", "$noun", "$past", "$capital", "$allcaps", "$only", "$onlys",
    "$stem", "$hasdot", "$sentence");
  private static final EspeakNgDictSource EMPTY = new EspeakNgDictSource(
    Map.of(), Map.of(), List.of(), Map.of(), List.of(), Set.of(), EspeakNgCompiledDict.none(),
    LanguageTables.NONE, false);

  private final Map<String, List<ListEntry>> listEntries;
  private final Map<String, List<PronRule>> ruleGroups;
  private final List<PronRule> defaultRules;
  private final Map<Integer, List<String>> letterGroups;
  private final List<String> replacements;
  private final Set<Integer> dictRules;
  private final EspeakNgCompiledDict compiled;
  private final LanguageTables languageTables;
  private final boolean loaded;

  private EspeakNgDictSource(
    final Map<String, List<ListEntry>> listEntries,
    final Map<String, List<PronRule>> ruleGroups,
    final List<PronRule> defaultRules,
    final Map<Integer, List<String>> letterGroups,
    final List<String> replacements,
    final Set<Integer> dictRules,
    final EspeakNgCompiledDict compiled,
    final LanguageTables languageTables,
    final boolean loaded
  ) {
    this.listEntries = Map.copyOf(listEntries);
    this.ruleGroups = Map.copyOf(ruleGroups);
    this.defaultRules = List.copyOf(defaultRules);
    this.letterGroups = Map.copyOf(letterGroups);
    this.replacements = List.copyOf(replacements);
    this.dictRules = Set.copyOf(dictRules);
    this.compiled = compiled;
    this.languageTables = languageTables;
    this.loaded = loaded;
  }

  private static Set<Integer> codePoints(final String letters) {
    return letters.codePoints().boxed().collect(toUnmodifiableSet());
  }

  private static String[] whitespaceTokens(final String text) {
    return WHITESPACE.split(text, -1);
  }

  static EspeakNgDictSource load(final Path dataDir, final String espeakVoice) {
    requireNonNull(dataDir, "dataDir");
    requireNonNull(espeakVoice, "espeakVoice");
    Path dictsource = dataDir.resolve("dictsource");
    Optional<EspeakNgVoiceFiles.Profile> profile = EspeakNgVoiceFiles.resolve(dataDir, espeakVoice);
    if (profile.isEmpty()) {
      return EMPTY;
    }
    if (Files.isDirectory(dictsource)) {
      Optional<Path> listx = dictionaryFile(profile.get(), dictsource, "_listx")
        .or(() -> dictionaryFile(profile.get(), dictsource.resolve("extra"), "_listx"));
      Optional<Path> list = dictionaryFile(profile.get(), dictsource, "_list");
      Optional<Path> extra = dictionaryFile(profile.get(), dictsource, "_extra");
      Optional<Path> rules = dictionaryFile(profile.get(), dictsource, "_rules");
      if (list.isPresent() || listx.isPresent() || rules.isPresent()) {
        try {
          Builder builder = new Builder(profile.get().dictRules());
          if (listx.isPresent()) {
            builder.readList(listx.get());
          }
          if (list.isPresent()) {
            builder.readList(list.get());
          }
          if (extra.isPresent()) {
            builder.readList(extra.get());
          }
          if (rules.isPresent()) {
            builder.readRules(rules.get());
          }
          return builder.build(compiledDictionary(dataDir, profile.get()));
        } catch (IOException ignored) {
          return EMPTY;
        }
      }
    }
    EspeakNgCompiledDict compiled = compiledDictionary(dataDir, profile.get());
    if (!compiled.isLoaded()) {
      return EMPTY;
    }
    return new EspeakNgDictSource(
      Map.of(), Map.of(), List.of(), Map.of(), List.of(), profile.get().dictRules(), compiled,
      LanguageTables.NONE, true);
  }

  private static Optional<Path> dictionaryFile(
    final EspeakNgVoiceFiles.Profile profile,
    final Path dir,
    final String suffix
  ) {
    if (!Files.isDirectory(dir)) {
      return Optional.empty();
    }
    return profile.dictionaryNames().stream()
      .map(name -> dir.resolve(name + suffix))
      .filter(Files::isRegularFile)
      .findFirst();
  }

  private static EspeakNgCompiledDict compiledDictionary(
    final Path dataDir,
    final EspeakNgVoiceFiles.Profile profile
  ) {
    return EspeakNgCompiledDict.load(
      dataDir,
      profile.dictionaryNames(),
      profile.phonemeTableNames(),
      profile.dictRules());
  }

  private static OptionalInt stressFlag(final String token) {
    return switch (token) {
      case "$1" -> OptionalInt.of(1);
      case "$2" -> OptionalInt.of(2);
      case "$3" -> OptionalInt.of(3);
      case "$4" -> OptionalInt.of(4);
      case "$5" -> OptionalInt.of(5);
      case "$6" -> OptionalInt.of(6);
      case "$7" -> OptionalInt.of(7);
      case "$u", "$u+" -> OptionalInt.of(-1);
      case "$u1", "$u1+" -> OptionalInt.of(1);
      case "$u2", "$u2+" -> OptionalInt.of(2);
      case "$u3", "$u3+" -> OptionalInt.of(3);
      default -> OptionalInt.empty();
    };
  }

  private static List<Atom> tokenize(final String pattern) {
    List<Atom> atoms = new ArrayList<>();
    int[] cps = pattern.codePoints().toArray();
    int i = 0;
    while (i < cps.length) {
      int cp = cps[i];
      if (cp == '_') {
        atoms.add(Atom.of(AtomKind.BOUNDARY));
        i++;
        continue;
      }
      if (cp == 'A') {
        atoms.add(Atom.of(AtomKind.VOWEL));
        i++;
        continue;
      }
      if (cp == 'C') {
        atoms.add(Atom.of(AtomKind.CONSONANT));
        i++;
        continue;
      }
      if (cp == 'K') {
        atoms.add(Atom.of(AtomKind.NOT_VOWEL));
        i++;
        continue;
      }
      if (cp == 'D') {
        atoms.add(Atom.of(AtomKind.DIGIT));
        i++;
        continue;
      }
      if (cp == 'Z') {
        atoms.add(Atom.of(AtomKind.NONALPHA));
        i++;
        continue;
      }
      if (cp == 'X') {
        atoms.add(Atom.of(AtomKind.NO_VOWELS));
        i++;
        continue;
      }
      if (cp == '%') {
        atoms.add(Atom.of(AtomKind.DOUBLE));
        i++;
        continue;
      }
      if (cp == '@' || cp == '&') {
        atoms.add(Atom.of(AtomKind.SYLLABLE));
        i++;
        continue;
      }
      if (cp == '+') {
        atoms.add(Atom.of(AtomKind.INC));
        i++;
        continue;
      }
      if (cp == '<') {
        atoms.add(Atom.of(AtomKind.DEC));
        i++;
        continue;
      }
      if (cp == '-') {
        atoms.add(Atom.of(AtomKind.HYPHEN));
        i++;
        continue;
      }
      if (cp == '/') {
        if (i + 1 < cps.length) {
          atoms.add(Atom.literal(cps[i + 1]));
          i += 2;
        } else {
          i++;
        }
        continue;
      }
      if (cp == 'L' && i + 2 < cps.length && Character.isDigit(cps[i + 1]) &&
        Character.isDigit(cps[i + 2])) {
        atoms.add(Atom.letterGroup((cps[i + 1] - '0') * 10 + (cps[i + 2] - '0')));
        i += 3;
        continue;
      }
      if (cp == 'Y' || cp == 'B' || cp == 'H' || cp == 'F' || cp == 'G') {
        atoms.add(Atom.namedGroup(cp));
        i++;
        continue;
      }
      if (cp == 'N') {
        atoms.add(Atom.of(AtomKind.NO_SUFFIX));
        i++;
        continue;
      }
      if ((cp == 'S' || cp == 'P') && i + 1 < cps.length && Character.isDigit(cps[i + 1])) {
        atoms.add(Atom.affix(parseAffix(cp, cps, i + 1)));
        break;
      }
      if (cp == '$' || cp == 'J' || cp == '#' || cp == 'V') {
        atoms.add(Atom.of(AtomKind.UNSUPPORTED));
        break;
      }
      atoms.add(Atom.literal(cp));
      i++;
    }
    return List.copyOf(atoms);
  }

  private static int parseAffix(final int kind, final int[] cps, final int start) {
    int i = start;
    int letters = 0;
    while (i < cps.length && Character.isDigit(cps[i])) {
      letters = letters * 10 + (cps[i] - '0');
      i++;
    }
    int flags = 0;
    while (i < cps.length) {
      int flag = switch (cps[i]) {
        case 'e' -> FLAG_E;
        case 'i' -> FLAG_I;
        case 'd' -> FLAG_D;
        case 'q' -> FLAG_Q;
        case 'm' -> FLAG_M;
        case 'v', 'f', 't', 'b', 'a', 'p' -> 0;
        default -> -1;
      };
      if (flag < 0) {
        break;
      }
      flags |= flag;
      i++;
    }
    return Affix.pack(kind == 'P' ? AffixKind.PREFIX : AffixKind.SUFFIX, letters, flags);
  }

  boolean isLoaded() {
    return this.loaded;
  }

  String toIpa(final String text) {
    String folded = text.strip().toLowerCase(ROOT);
    if (folded.isEmpty() || !this.loaded) {
      return "";
    }
    StringBuilder ipa = new StringBuilder(folded.length() * 2);
    for (String word : WORD_BREAK.split(this.applyReplace(folded), -1)) {
      if (word.isEmpty()) {
        continue;
      }
      String letters = this.stripPunctuation(word);
      if (letters.isEmpty()) {
        continue;
      }
      Optional<String> phonemes = this.phonemesForWord(letters, 0, false, false);
      if (phonemes.isEmpty()) {
        continue;
      }
      String wordIpa = EspeakNgPhonemes.toIpa(phonemes.get());
      if (wordIpa.isEmpty()) {
        continue;
      }
      if (!ipa.isEmpty()) {
        ipa.append(' ');
      }
      ipa.append(wordIpa);
    }
    return ipa.toString();
  }

  Optional<String> phonemesForWord(final String word) {
    return this.phonemesForWord(word, 0, false, false);
  }

  private Optional<String> phonemesForWord(
    final String word,
    final int depth,
    final boolean suffixRemoved,
    final boolean prefixRemoved
  ) {
    String rewritten = this.applyReplace(word);
    if (rewritten.isEmpty()) {
      return Optional.empty();
    }
    Optional<String> number = this.translateNumber(rewritten);
    if (number.isPresent()) {
      return number;
    }
    Optional<ListEntry> listed = this.matchList(rewritten);
    if (listed.isPresent() && !listed.get().phonemes().isEmpty()) {
      return Optional.of(listed.get().phonemes());
    }
    Optional<EspeakNgCompiledDict.Hit> compiled = this.compiled.lookupHit(rewritten);
    if (compiled.isPresent() && compiled.get().hasPhonemes()) {
      return Optional.of(compiled.get().phonemes());
    }
    if (this.ruleGroups.isEmpty() && this.defaultRules.isEmpty()) {
      return Optional.empty();
    }
    Optional<String> byRules =
      this.translateByRules(rewritten, depth, suffixRemoved, prefixRemoved);
    if (depth > 0 || byRules.isEmpty()) {
      return byRules;
    }
    int stress = listed.map(ListEntry::stressSyllable)
      .filter(value -> value != 0)
      .orElseGet(() -> compiled.map(EspeakNgCompiledDict.Hit::stressSyllable).orElse(0));
    return Optional.of(this.applyLanguagePhonology(byRules.get(), stress));
  }

  private String applyLanguagePhonology(final String phonemes, final int stressSyllable) {
    return this.languageTables.russianPhonology()
      ? EspeakNgRussianPhonology.apply(phonemes, stressSyllable)
      : phonemes;
  }

  private Optional<String> translateByRules(
    final String word,
    final int depth,
    final boolean suffixRemoved,
    final boolean prefixRemoved
  ) {
    RuleOutcome outcome = this.applyRules(word, suffixRemoved);
    if (depth >= MAX_AFFIX_DEPTH) {
      return outcome.phonemes().isEmpty() ? Optional.empty() : Optional.of(outcome.phonemes());
    }
    if (outcome.affix().kind() == AffixKind.PREFIX && !prefixRemoved) {
      return this.applyPrefix(word, outcome, depth, suffixRemoved);
    }
    if (outcome.affix().kind() == AffixKind.SUFFIX) {
      return this.applySuffix(word, outcome, depth, prefixRemoved);
    }
    return outcome.phonemes().isEmpty() ? Optional.empty() : Optional.of(outcome.phonemes());
  }

  private Optional<String> applyPrefix(
    final String word,
    final RuleOutcome outcome,
    final int depth,
    final boolean suffixRemoved
  ) {
    int letters = Math.min(outcome.affix().letters(), word.codePointCount(0, word.length()));
    if (letters <= 0 || letters >= word.codePointCount(0, word.length())) {
      return outcome.phonemes().isEmpty() ? Optional.empty() : Optional.of(outcome.phonemes());
    }
    int split = word.offsetByCodePoints(0, letters);
    String rest = word.substring(split);
    Optional<String> stem = this.phonemesForWord(rest, depth + 1, suffixRemoved, true);
    String joined = outcome.affixPhonemes() + stem.orElse("");
    return joined.isEmpty() ? Optional.empty() : Optional.of(joined);
  }

  private Optional<String> applySuffix(
    final String word,
    final RuleOutcome outcome,
    final int depth,
    final boolean prefixRemoved
  ) {
    if ((outcome.affix().flags() & FLAG_Q) != 0) {
      String joined = outcome.phonemes() + outcome.affixPhonemes();
      return joined.isEmpty() ? Optional.empty() : Optional.of(joined);
    }
    String stem = this.removeEnding(word, outcome.affix());
    if (stem.isEmpty() || stem.equals(word)) {
      String joined = outcome.phonemes() + outcome.affixPhonemes();
      return joined.isEmpty() ? Optional.empty() : Optional.of(joined);
    }
    Optional<String> stemPhonemes = this.phonemesForWord(stem, depth + 1, true, prefixRemoved);
    if (stemPhonemes.isEmpty() && (outcome.affix().flags() & FLAG_D) != 0 &&
      this.hasDoubleFinal(stem)) {
      stemPhonemes = this.phonemesForWord(this.undoubleFinal(stem), depth + 1, true, prefixRemoved);
    }
    String joined = stemPhonemes.orElse("") + outcome.affixPhonemes();
    return joined.isEmpty() ? Optional.empty() : Optional.of(joined);
  }

  private String removeEnding(final String word, final Affix affix) {
    int letters = word.codePointCount(0, word.length());
    int strip = Math.min(affix.letters(), letters);
    if (strip <= 0) {
      return word;
    }
    int end = word.offsetByCodePoints(0, letters - strip);
    String stem = word.substring(0, end);
    if ((affix.flags() & FLAG_I) != 0 && stem.endsWith("i")) {
      stem = stem.substring(0, stem.length() - 1) + "y";
    }
    if ((affix.flags() & FLAG_E) != 0 && this.shouldAddE(stem)) {
      stem = stem + "e";
    }
    return stem;
  }

  private boolean shouldAddE(final String stem) {
    if (stem.isEmpty()) {
      return false;
    }
    for (String exception : ADD_E_EXCEPTIONS) {
      if (stem.endsWith(exception)) {
        return false;
      }
    }
    int last = stem.codePointBefore(stem.length());
    if (this.isConsonant(last) && stem.length() >= 2) {
      int before =
        stem.codePointBefore(stem.offsetByCodePoints(0, stem.codePointCount(0, stem.length()) - 1));
      if (this.isVowel(before) || before == 'y' || before == 'Y') {
        return true;
      }
    }
    return ADD_E_ADDITIONS.stream().anyMatch(stem::endsWith);
  }

  private boolean hasDoubleFinal(final String word) {
    int letters = word.codePointCount(0, word.length());
    if (letters < 2) {
      return false;
    }
    int last = word.codePointBefore(word.length());
    int prevIndex = word.offsetByCodePoints(0, letters - 1);
    return word.codePointBefore(prevIndex) == last;
  }

  private String undoubleFinal(final String word) {
    return word.substring(0, word.offsetByCodePoints(0, word.codePointCount(0, word.length()) - 1));
  }

  private Optional<String> translateNumber(final String word) {
    if (!DIGITS.matcher(word).matches()) {
      return Optional.empty();
    }
    int decimalAt = Math.max(word.indexOf('.'), word.indexOf(','));
    String integer = decimalAt < 0 ? word : word.substring(0, decimalAt);
    String fraction = decimalAt < 0 ? "" : word.substring(decimalAt + 1);
    if (integer.startsWith("0") && integer.length() > 1 && fraction.isEmpty()) {
      return this.speakDigits(integer);
    }
    if (integer.length() > 12) {
      return this.speakDigits(integer);
    }
    String spoken = this.speakInteger(this.parseUnsigned(integer));
    if (!fraction.isEmpty()) {
      spoken =
        spoken + this.lookupFragment("_dpt").orElse("") + this.speakDigits(fraction).orElse("");
    }
    return spoken.isEmpty() ? Optional.empty() : Optional.of(spoken);
  }

  private long parseUnsigned(final String digits) {
    if (digits.isEmpty()) {
      return 0;
    }
    try {
      return Long.parseLong(digits);
    } catch (NumberFormatException ignored) {
      return 0;
    }
  }

  private Optional<String> speakDigits(final String digits) {
    StringBuilder phonemes = new StringBuilder();
    for (int i = 0; i < digits.length(); i++) {
      phonemes.append(this.lookupFragment("_" + digits.charAt(i)).orElse(""));
    }
    return phonemes.isEmpty() ? Optional.empty() : Optional.of(phonemes.toString());
  }

  private String speakInteger(final long value) {
    if (value == 0) {
      return this.lookupFragment("_0").orElse("");
    }
    StringBuilder phonemes = new StringBuilder();
    long remaining = value;
    int scale = 0;
    long[] groups = new long[8];
    while (remaining > 0 && scale < groups.length) {
      groups[scale] = remaining % 1000;
      remaining /= 1000;
      scale++;
    }
    for (int i = scale - 1; i >= 0; i--) {
      if (groups[i] == 0) {
        continue;
      }
      boolean needAnd = i == 0 && !phonemes.isEmpty() && groups[i] < 100;
      phonemes.append(this.speakTriple((int) groups[i], needAnd));
      if (i > 0) {
        phonemes.append(this.lookupFragment("_0M" + i).orElse(""));
      }
    }
    return phonemes.toString();
  }

  private String speakTriple(final int value, final boolean andAfterHundred) {
    int hundreds = value / 100;
    int rest = value % 100;
    StringBuilder phonemes = new StringBuilder();
    if (hundreds > 0) {
      Optional<String> special = this.lookupFragment("_" + hundreds + "C");
      if (rest == 0) {
        Optional<String> exact = this.lookupFragment("_" + hundreds + "C0");
        if (exact.isPresent()) {
          return exact.get();
        }
      }
      phonemes.append(special.orElseGet(() -> this.lookupFragment("_" + hundreds).orElse("")));
      if (special.isEmpty()) {
        phonemes.append(this.lookupFragment("_0C").orElse(""));
      }
    }
    if (rest > 0) {
      if (hundreds > 0 || andAfterHundred) {
        phonemes.append(this.lookupFragment("_0and").orElse(""));
      }
      phonemes.append(this.speakBelowHundred(rest));
    }
    return phonemes.toString();
  }

  private String speakBelowHundred(final int value) {
    Optional<String> exact = this.lookupFragment("_" + value);
    if (exact.isPresent()) {
      return exact.get();
    }
    int tens = value / 10;
    int ones = value % 10;
    if (tens >= 2) {
      String spoken = this.lookupFragment("_" + tens + "X").orElse("");
      if (ones > 0) {
        spoken = spoken + this.lookupFragment("_" + ones).orElse("");
      }
      return spoken;
    }
    return this.lookupFragment("_" + ones).orElse("");
  }

  private Optional<String> lookupFragment(final String key) {
    String folded = key.toLowerCase(ROOT);
    return this.lookupList(folded).or(() -> this.compiled.lookup(folded));
  }

  private String applyReplace(final String text) {
    String rewritten = text;
    for (int i = 0; i + 1 < this.replacements.size(); i += 2) {
      rewritten = rewritten.replace(this.replacements.get(i), this.replacements.get(i + 1));
    }
    return rewritten;
  }

  private String stripPunctuation(final String word) {
    StringBuilder letters = new StringBuilder(word.length());
    word.codePoints()
      .filter(cp -> Character.isLetter(cp) || Character.isDigit(cp) || cp == '\'' || cp == '.' ||
        cp == ',')
      .forEach(letters::appendCodePoint);
    return letters.toString();
  }

  private Optional<String> lookupList(final String word) {
    return this.matchList(word)
      .map(ListEntry::phonemes)
      .filter(phonemes -> !phonemes.isEmpty());
  }

  private Optional<ListEntry> matchList(final String word) {
    List<ListEntry> entries = this.listEntries.get(word);
    if (entries == null) {
      return Optional.empty();
    }
    for (int i = entries.size() - 1; i >= 0; i--) {
      ListEntry entry = entries.get(i);
      if (!this.conditionMatches(entry.condition(), entry.negated())) {
        continue;
      }
      if (entry.phonemes().isEmpty() && entry.stressSyllable() == 0) {
        continue;
      }
      if (entry.textMode()) {
        return this.matchList(entry.phonemes()).or(() -> this.phonemesForWord(entry.phonemes())
          .map(phonemes -> new ListEntry(phonemes, false, 0, false, 0)));
      }
      return Optional.of(entry);
    }
    return Optional.empty();
  }

  private boolean conditionMatches(final int condition, final boolean negated) {
    if (condition <= 0) {
      return true;
    }
    boolean present = this.dictRules.contains(condition);
    return negated != present;
  }

  private RuleOutcome applyRules(final String word, final boolean suffixRemoved) {
    int[] padded = this.padded(word);
    StringBuilder phonemes = new StringBuilder(word.length() * 2);
    int index = 1;
    int end = padded.length - 1;
    while (index < end) {
      BestMatch best = this.bestRule(padded, index, end, suffixRemoved);
      if (best.score() <= 0 || best.consumed() <= 0) {
        index++;
        continue;
      }
      if (best.affix().kind() != AffixKind.NONE) {
        return new RuleOutcome(phonemes.toString(), best.affix(), best.phonemes());
      }
      phonemes.append(best.phonemes());
      index += best.consumed();
    }
    return new RuleOutcome(phonemes.toString(), Affix.NONE, "");
  }

  private int[] padded(final String word) {
    int[] letters = word.codePoints().toArray();
    int[] padded = new int[letters.length + 2];
    padded[0] = ' ';
    System.arraycopy(letters, 0, padded, 1, letters.length);
    padded[padded.length - 1] = ' ';
    return padded;
  }

  private BestMatch bestRule(final int[] word, final int index, final int end,
                             final boolean suffixRemoved) {
    BestMatch best = BestMatch.NONE;
    if (index + 1 < end) {
      String two = new String(word, index, 2);
      best = this.better(best, this.matchGroup(two, word, index, 2, suffixRemoved));
    }
    String one = new String(word, index, 1);
    best = this.better(best, this.matchGroup(one, word, index, 1, suffixRemoved));
    return this.better(best, this.matchRules(this.defaultRules, word, index, 0, suffixRemoved));
  }

  private BestMatch matchGroup(
    final String group,
    final int[] word,
    final int index,
    final int groupLength,
    final boolean suffixRemoved
  ) {
    List<PronRule> rules = this.ruleGroups.get(group);
    if (rules == null) {
      return BestMatch.NONE;
    }
    BestMatch best = this.matchRules(rules, word, index, groupLength, suffixRemoved);
    if (groupLength > 1 && best.score() > 0) {
      return new BestMatch(best.score() + 35, best.consumed(), best.phonemes(), best.affix());
    }
    return best;
  }

  private BestMatch matchRules(
    final List<PronRule> rules,
    final int[] word,
    final int index,
    final int groupLength,
    final boolean suffixRemoved
  ) {
    BestMatch best = BestMatch.NONE;
    for (PronRule rule : rules) {
      if (!this.conditionMatches(rule.condition(), rule.negated())) {
        continue;
      }
      BestMatch matched = this.matchRule(rule, word, index, groupLength, suffixRemoved);
      best = this.better(best, matched);
    }
    return best;
  }

  private BestMatch matchRule(
    final PronRule rule,
    final int[] word,
    final int index,
    final int groupLength,
    final boolean suffixRemoved
  ) {
    int[] match = rule.match();
    if (match.length < groupLength || !this.startsWith(word, index, match)) {
      return BestMatch.NONE;
    }
    int score = 1 + 21 * Math.max(0, match.length - groupLength);
    int preScore = this.matchAtoms(word, index, rule.pre(), false, match, suffixRemoved);
    if (preScore < 0) {
      return BestMatch.NONE;
    }
    int postScore =
      this.matchAtoms(word, index + match.length, rule.post(), true, match, suffixRemoved);
    if (postScore < 0) {
      return BestMatch.NONE;
    }
    return new BestMatch(
      score + preScore + postScore, match.length, rule.phonemes(), this.affixOf(rule.post()));
  }

  private int matchAtoms(
    final int[] word,
    final int origin,
    final List<Atom> atoms,
    final boolean forward,
    final int[] matchLetters,
    final boolean suffixRemoved
  ) {
    int cursor = origin;
    int lastLetter = matchLetters.length == 0
      ? 0
      : (forward ? matchLetters[matchLetters.length - 1] : matchLetters[0]);
    int score = 0;
    int distance = forward ? -6 : -2;
    int from = forward ? 0 : atoms.size() - 1;
    int step = forward ? 1 : -1;
    for (int i = from; forward ? i < atoms.size() : i >= 0; i += step) {
      distance += forward ? 6 : 2;
      if (distance > 18) {
        distance = 19;
      }
      Atom atom = atoms.get(i);
      if (atom.kind() == AtomKind.SYLLABLE) {
        int run = 1;
        int look = i + step;
        while (forward ? look < atoms.size() : look >= 0) {
          if (atoms.get(look).kind() != AtomKind.SYLLABLE) {
            break;
          }
          run++;
          look += step;
        }
        i = look - step;
        if (this.vowelCount(word, cursor, forward) < run) {
          return -1;
        }
        score += 18 + run;
        continue;
      }
      int points = this.matchAtom(atom, word, cursor, lastLetter, forward, distance, suffixRemoved);
      if (points == Integer.MIN_VALUE) {
        return -1;
      }
      if (atom.kind() != AtomKind.INC && atom.kind() != AtomKind.DEC &&
        atom.kind() != AtomKind.NO_VOWELS
        && atom.kind() != AtomKind.AFFIX && atom.kind() != AtomKind.NO_SUFFIX) {
        int consumed = this.atomWidth(atom, word, cursor, forward);
        if (consumed > 0) {
          int nextIndex = forward ? cursor : cursor - consumed;
          if (nextIndex >= 0 && nextIndex < word.length) {
            lastLetter = word[nextIndex];
          }
          cursor = forward ? cursor + consumed : cursor - consumed;
        }
      }
      score += points;
    }
    return score;
  }

  private int matchAtom(
    final Atom atom,
    final int[] word,
    final int cursor,
    final int lastLetter,
    final boolean forward,
    final int distance,
    final boolean suffixRemoved
  ) {
    int probe = this.letterAt(word, cursor, forward);
    return switch (atom.kind()) {
      case LITERAL -> probe == atom.value()
        ? (probe == ' ' ? 4 : Math.max(1, 21 - distance))
        : Integer.MIN_VALUE;
      case BOUNDARY -> probe == ' ' ? 4 : Integer.MIN_VALUE;
      case HYPHEN ->
        (probe == '-' || probe == ' ') ? Math.max(1, 22 - distance) : Integer.MIN_VALUE;
      case VOWEL -> this.isVowel(probe) ? Math.max(1, 20 - distance) : Integer.MIN_VALUE;
      case CONSONANT -> this.isConsonant(probe)
        ? Math.max(1, 19 - distance)
        : Integer.MIN_VALUE;
      case NOT_VOWEL -> !this.isVowel(probe) ? Math.max(1, 20 - distance) : Integer.MIN_VALUE;
      case DIGIT -> Character.isDigit(probe) ? Math.max(1, 20 - distance) : Integer.MIN_VALUE;
      case NONALPHA -> !Character.isLetter(probe) ? Math.max(1, 21 - distance) : Integer.MIN_VALUE;
      case DOUBLE -> probe == lastLetter ? Math.max(1, 21 - distance) : Integer.MIN_VALUE;
      case LETTER_GROUP -> this.letterGroupWidth(word, cursor, atom.value(), forward) >= 0
        ? Math.max(1, 20 - distance)
        : Integer.MIN_VALUE;
      case NAMED_LETTERGP -> this.inNamedGroup(probe, atom.value())
        ? Math.max(1, 20 - distance)
        : Integer.MIN_VALUE;
      case NO_VOWELS -> this.noVowelUntilBoundary(word, cursor, forward)
        ? Math.max(1, 19 - distance)
        : Integer.MIN_VALUE;
      case SYLLABLE -> this.vowelCount(word, cursor, forward) >= 1 ? 18 : Integer.MIN_VALUE;
      case INC -> 20;
      case DEC -> -20;
      case AFFIX -> 1;
      case NO_SUFFIX -> suffixRemoved ? Integer.MIN_VALUE : 1;
      case UNSUPPORTED -> Integer.MIN_VALUE;
    };
  }

  private int atomWidth(final Atom atom, final int[] word, final int cursor,
                        final boolean forward) {
    return switch (atom.kind()) {
      case LITERAL, BOUNDARY, HYPHEN, VOWEL, CONSONANT, NOT_VOWEL, DIGIT, NONALPHA,
           DOUBLE, NAMED_LETTERGP -> 1;
      case LETTER_GROUP -> Math.max(0, this.letterGroupWidth(word, cursor, atom.value(), forward));
      default -> 0;
    };
  }

  private boolean inNamedGroup(final int probe, final int group) {
    Set<Integer> letters = this.languageTables.namedLetterGroups().get(group);
    if (letters != null) {
      return letters.contains(probe);
    }
    if (group == 'Y') {
      return probe == 'y' || probe == 'Y';
    }
    return this.isConsonant(probe);
  }

  private int letterAt(final int[] word, final int cursor, final boolean forward) {
    int index = forward ? cursor : cursor - 1;
    if (index < 0 || index >= word.length) {
      return 0;
    }
    return word[index];
  }

  private int letterGroupWidth(
    final int[] word,
    final int cursor,
    final int group,
    final boolean forward
  ) {
    List<String> items = this.letterGroups.getOrDefault(group, List.of());
    for (String item : items) {
      if (item.isEmpty()) {
        return 0;
      }
      int[] expected = item.codePoints().toArray();
      if (forward) {
        if (this.startsWith(word, cursor, expected)) {
          return expected.length;
        }
      } else if (this.endsWith(word, cursor, expected)) {
        return expected.length;
      }
    }
    return -1;
  }

  private boolean startsWith(final int[] word, final int index, final int[] expected) {
    if (index + expected.length > word.length) {
      return false;
    }
    for (int i = 0; i < expected.length; i++) {
      if (word[index + i] != expected[i]) {
        return false;
      }
    }
    return true;
  }

  private boolean endsWith(final int[] word, final int endExclusive, final int[] expected) {
    int start = endExclusive - expected.length;
    return start >= 0 && this.startsWith(word, start, expected);
  }

  private boolean noVowelUntilBoundary(final int[] word, final int cursor, final boolean forward) {
    int i = forward ? cursor : cursor - 1;
    int step = forward ? 1 : -1;
    while (i >= 0 && i < word.length && word[i] != ' ') {
      if (this.isVowel(word[i])) {
        return false;
      }
      i += step;
    }
    return true;
  }

  private int vowelCount(final int[] word, final int cursor, final boolean forward) {
    int i = forward ? cursor : cursor - 1;
    int step = forward ? 1 : -1;
    int count = 0;
    while (i >= 0 && i < word.length && word[i] != ' ') {
      if (this.isVowel(word[i])) {
        count++;
      }
      i += step;
    }
    return count;
  }

  private Affix affixOf(final List<Atom> post) {
    return post.stream()
      .filter(atom -> atom.kind() == AtomKind.AFFIX)
      .reduce((left, right) -> right)
      .map(atom -> Affix.decode(atom.value()))
      .orElse(Affix.NONE);
  }

  private boolean isVowel(final int cp) {
    return LATIN_VOWELS.indexOf(cp) >= 0 || CYRILLIC_VOWELS.indexOf(cp) >= 0;
  }

  private boolean isConsonant(final int cp) {
    return Character.isLetter(cp) && !this.isVowel(cp);
  }

  private BestMatch better(final BestMatch left, final BestMatch right) {
    return right.score() >= left.score() ? right : left;
  }

  private enum AtomKind {
    LITERAL, BOUNDARY, VOWEL, CONSONANT, NOT_VOWEL, DIGIT, NONALPHA, NO_VOWELS, DOUBLE, SYLLABLE,
    LETTER_GROUP, NAMED_LETTERGP, INC, DEC, HYPHEN, AFFIX, NO_SUFFIX, UNSUPPORTED
  }

  private enum AffixKind {
    NONE, SUFFIX, PREFIX
  }

  private record ListEntry(
    String phonemes,
    boolean textMode,
    int condition,
    boolean negated,
    int stressSyllable
  ) {
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record PronRule(
    List<Atom> pre,
    int[] match,
    List<Atom> post,
    String phonemes,
    int condition,
    boolean negated
  ) {
  }

  private record Atom(AtomKind kind, int value) {
    static Atom of(final AtomKind kind) {
      return new Atom(kind, 0);
    }

    static Atom literal(final int cp) {
      return new Atom(AtomKind.LITERAL, cp);
    }

    static Atom namedGroup(final int group) {
      return new Atom(AtomKind.NAMED_LETTERGP, group);
    }

    static Atom letterGroup(final int group) {
      return new Atom(AtomKind.LETTER_GROUP, group);
    }

    static Atom affix(final int packed) {
      return new Atom(AtomKind.AFFIX, packed);
    }
  }

  private record BestMatch(int score, int consumed, String phonemes, Affix affix) {
    static final BestMatch NONE = new BestMatch(0, 0, "", Affix.NONE);
  }

  private record RuleOutcome(String phonemes, Affix affix, String affixPhonemes) {
  }

  private record Affix(AffixKind kind, int letters, int flags) {
    static final Affix NONE = new Affix(AffixKind.NONE, 0, 0);

    static Affix decode(final int packed) {
      AffixKind kind = (packed & 1) != 0 ? AffixKind.PREFIX : AffixKind.SUFFIX;
      return new Affix(kind, (packed >>> 1) & 0x3f, packed >>> 8);
    }

    static int pack(final AffixKind kind, final int letters, final int flags) {
      return (kind == AffixKind.PREFIX ? 1 : 0) | ((letters & 0x3f) << 1) | (flags << 8);
    }
  }

  private static final class Builder {

    private final Map<String, List<ListEntry>> listEntries = new HashMap<>();
    private final Map<String, List<PronRule>> ruleGroups = new HashMap<>();
    private final List<PronRule> defaultRules = new ArrayList<>();
    private final Map<Integer, List<String>> letterGroups = new HashMap<>();
    private final List<String> replacements = new ArrayList<>();
    private final Set<Integer> dictRules;
    private boolean textMode;
    private String currentGroup = "";
    private boolean replacing;

    Builder(final Set<Integer> dictRules) {
      this.dictRules = Set.copyOf(dictRules);
    }

    void readList(final Path file) throws IOException {
      for (String raw : Files.readAllLines(file, UTF_8)) {
        this.readListLine(this.stripComment(raw).strip());
      }
    }

    void readRules(final Path file) throws IOException {
      for (String raw : Files.readAllLines(file, UTF_8)) {
        this.readRuleLine(this.stripComment(raw).strip());
      }
    }

    EspeakNgDictSource build(final EspeakNgCompiledDict compiled) {
      requireNonNull(compiled, "compiled");
      this.letterGroups.replaceAll((key, items) -> items.stream()
        .sorted(comparingInt(String::length).reversed())
        .toList());
      return new EspeakNgDictSource(
        this.listEntries,
        this.ruleGroups,
        this.defaultRules,
        this.letterGroups,
        this.replacements,
        this.dictRules,
        compiled,
        LanguageTables.detect(this.ruleGroups.keySet()),
        true);
    }

    private void readListLine(final String line) {
      if (line.isEmpty()) {
        return;
      }
      if (line.equals("$textmode")) {
        this.textMode = true;
        return;
      }
      if (line.equals("$phonememode")) {
        this.textMode = false;
        return;
      }
      Condition condition = Condition.parse(line);
      String rest = condition.rest();
      if (rest.startsWith("(")) {
        return;
      }
      String[] parts = whitespaceTokens(rest);
      if (parts.length == 0 || parts[0].isEmpty()) {
        return;
      }
      if (rest.startsWith("_") && !NUMBER_KEY.matcher(parts[0]).matches()) {
        return;
      }
      String word = parts[0].toLowerCase(ROOT);
      boolean text = this.textMode;
      String phonemes = "";
      int stress = 0;
      for (int i = 1; i < parts.length; i++) {
        String token = parts[i];
        if (CONTEXT_FLAGS.contains(token)) {
          return;
        }
        OptionalInt flagged = stressFlag(token);
        if (flagged.isPresent()) {
          stress = flagged.getAsInt();
          continue;
        }
        if (token.equals("$text")) {
          text = true;
          continue;
        }
        if (token.startsWith("$")) {
          continue;
        }
        phonemes = token;
      }
      if (phonemes.isEmpty() && stress == 0) {
        return;
      }
      this.listEntries
        .computeIfAbsent(word, key -> new ArrayList<>())
        .add(new ListEntry(phonemes, text, condition.number(), condition.negated(), stress));
    }

    private void readRuleLine(final String line) {
      if (line.isEmpty()) {
        return;
      }
      if (line.equals(".replace")) {
        this.replacing = true;
        this.currentGroup = "";
        return;
      }
      if (line.startsWith(".L")) {
        this.replacing = false;
        this.readLetterGroup(line);
        return;
      }
      if (line.startsWith(".group")) {
        this.replacing = false;
        this.currentGroup = line.substring(".group".length()).strip();
        return;
      }
      if (line.startsWith(".")) {
        this.replacing = false;
        return;
      }
      if (this.replacing) {
        this.readReplace(line);
        return;
      }
      Condition condition = Condition.parse(line);
      Optional<PronRule> rule =
        this.parseRule(condition.rest(), condition.number(), condition.negated());
      if (rule.isEmpty()) {
        return;
      }
      if (this.currentGroup.isEmpty()) {
        this.defaultRules.add(rule.get());
        return;
      }
      this.ruleGroups.computeIfAbsent(this.currentGroup, key -> new ArrayList<>()).add(rule.get());
    }

    private void readLetterGroup(final String line) {
      if (line.length() < 4) {
        return;
      }
      String id = line.substring(2, 4);
      if (!Character.isDigit(id.charAt(0)) || !Character.isDigit(id.charAt(1))) {
        return;
      }
      int group = Integer.parseInt(id);
      List<String> items = new ArrayList<>();
      for (String token : whitespaceTokens(line.substring(4).strip())) {
        if (token.isEmpty()) {
          continue;
        }
        items.add(token.equals("~") ? "" : token);
      }
      this.letterGroups.put(group, items);
    }

    private void readReplace(final String line) {
      String[] parts = whitespaceTokens(line);
      if (parts.length < 2) {
        return;
      }
      this.replacements.add(parts[0]);
      this.replacements.add(parts[1]);
    }

    private Optional<PronRule> parseRule(final String line, final int condition,
                                         final boolean negated) {
      int close = line.indexOf(')');
      String preText = "";
      String rest = line;
      if (close >= 0) {
        preText = line.substring(0, close).strip();
        rest = line.substring(close + 1).strip();
      }
      String matchText;
      String postText = "";
      String phonemes;
      int postAt = rest.indexOf(" (");
      if (postAt >= 0) {
        matchText = rest.substring(0, postAt).strip();
        String after = rest.substring(postAt + 2);
        int space = after.indexOf(' ');
        if (space < 0) {
          postText = after.strip();
          phonemes = "";
        } else {
          postText = after.substring(0, space);
          phonemes = after.substring(space + 1).strip();
        }
      } else {
        int space = rest.indexOf(' ');
        if (space < 0) {
          matchText = rest;
          phonemes = "";
        } else {
          matchText = rest.substring(0, space);
          phonemes = rest.substring(space + 1).strip();
        }
      }
      if (matchText.isEmpty()) {
        return Optional.empty();
      }
      List<Atom> pre = tokenize(preText);
      List<Atom> post = tokenize(postText);
      if (this.hasUnsupported(pre) || this.hasUnsupported(post)) {
        return Optional.empty();
      }
      return Optional.of(new PronRule(
        pre,
        matchText.codePoints().toArray(),
        post,
        whitespaceTokens(phonemes)[0],
        condition,
        negated));
    }

    private boolean hasUnsupported(final List<Atom> atoms) {
      return atoms.stream().anyMatch(atom -> atom.kind() == AtomKind.UNSUPPORTED);
    }

    private String stripComment(final String line) {
      int comment = line.indexOf("//");
      return comment < 0 ? line : line.substring(0, comment);
    }
  }

  private record Condition(int number, boolean negated, String rest) {

    static Condition parse(final String line) {
      if (!line.startsWith("?")) {
        return new Condition(0, false, line);
      }
      boolean negated = line.startsWith("?!");
      int start = negated ? 2 : 1;
      int end = start;
      while (end < line.length() && Character.isDigit(line.charAt(end))) {
        end++;
      }
      if (end == start) {
        return new Condition(0, false, line);
      }
      return new Condition(
        Integer.parseInt(line.substring(start, end)),
        negated,
        line.substring(end).strip());
    }
  }

  private record LanguageTables(
    Map<Integer, Set<Integer>> namedLetterGroups,
    boolean russianPhonology
  ) {
    static final LanguageTables NONE = new LanguageTables(Map.of(), false);

    LanguageTables {
      namedLetterGroups = Map.copyOf(requireNonNull(namedLetterGroups, "namedLetterGroups"));
    }

    static LanguageTables detect(final Set<String> ruleGroups) {
      boolean russian = ruleGroups.stream().anyMatch(LanguageTables::isCyrillic);
      return russian ? new LanguageTables(RUSSIAN_LETTER_GROUPS, true) : NONE;
    }

    private static boolean isCyrillic(final String group) {
      return group.codePoints().anyMatch(cp -> Character.UnicodeScript.of(cp) ==
        Character.UnicodeScript.CYRILLIC);
    }
  }
}
