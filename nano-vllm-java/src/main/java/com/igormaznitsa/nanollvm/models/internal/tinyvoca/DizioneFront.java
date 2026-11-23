package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.models.internal.tinyvoca.TinyVocaText.Phone;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * English front end: {@code phonemizer/lexicon.tsv} (word, then space-separated IPA with
 * {@code ˈ} / {@code ˌ}). A word missing from the lexicon is spelled letter by letter.
 * The neural G2P and the homograph tables are not in this port.
 */
public final class DizioneFront implements TinyVocaFront {

  private static final Map<String, String[]> LETTERS = letterNames();
  private static final Map<String, String[]> FALLBACK = fallback();
  private static final String[] SMALL = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
  };
  private static final String[] TENS = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
  };
  private final TinyVocaSpec spec;
  private final Map<String, String> lexicon;
  private final List<String> symbolsByLength;

  private DizioneFront(final TinyVocaSpec spec, final Map<String, String> lexicon) {
    this.spec = spec;
    this.lexicon = lexicon;
    this.symbolsByLength = spec.symbols().stream()
      .sorted(Comparator.comparingInt(String::length).reversed())
      .toList();
  }

  public static DizioneFront load(final TinyVocaSpec spec, final Path modelDir) throws IOException {
    requireNonNull(spec, "spec");
    requireNonNull(modelDir, "modelDir");
    Path lexicon = lexiconFile(modelDir);
    if (lexicon == null) {
      throw new IllegalArgumentException(
        "English tiny-voca needs phonemizer/lexicon.tsv next to the checkpoint. "
          + "Run models/download-tiny-voca.sh");
    }
    return fromLexicon(spec, Files.readString(lexicon, UTF_8));
  }

  public static DizioneFront fromLexicon(final TinyVocaSpec spec, final String text) {
    requireNonNull(spec, "spec");
    requireNonNull(text, "text");
    Map<String, String> words = new LinkedHashMap<>();
    for (String line : text.split("\\R")) {
      int tab = line.indexOf('\t');
      if (tab <= 0 || line.startsWith("#")) {
        continue;
      }
      words.put(line.substring(0, tab).toLowerCase(Locale.ROOT), line.substring(tab + 1).strip());
    }
    if (words.isEmpty()) {
      throw new IllegalArgumentException("phonemizer lexicon is empty");
    }
    return new DizioneFront(spec, words);
  }

  static boolean lexiconFileMissing(final Path modelDir) {
    return modelDir == null || lexiconFile(modelDir) == null;
  }

  private static Path lexiconFile(final Path modelDir) {
    Path parent = modelDir.getParent();
    Path[] candidates = {
      modelDir.resolve("phonemizer").resolve("lexicon.tsv"),
      modelDir.resolve("lexicon.tsv"),
      parent == null ? null : parent.resolve("phonemizer").resolve("lexicon.tsv")
    };
    for (Path candidate : candidates) {
      if (candidate != null && Files.isRegularFile(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  private static String sayNumbers(final String text) {
    String times = text.replaceAll("\\b(\\d{1,2}):(\\d{2})\\b", "$1 $2");
    StringBuilder out = new StringBuilder();
    int index = 0;
    while (index < times.length()) {
      char ch = times.charAt(index);
      if (!Character.isDigit(ch)) {
        out.append(ch);
        index++;
        continue;
      }
      int end = index;
      while (end < times.length() && Character.isDigit(times.charAt(end))) {
        end++;
      }
      out.append(cardinal(Long.parseLong(times.substring(index, Math.min(end, index + 6)))));
      index = end;
    }
    return out.toString();
  }

  private static String cardinal(final long value) {
    if (value < 20) {
      return SMALL[(int) value];
    }
    if (value < 100) {
      return TENS[(int) (value / 10)] + (value % 10 == 0 ? "" : " " + SMALL[(int) (value % 10)]);
    }
    if (value < 1000) {
      return SMALL[(int) (value / 100)] + " hundred"
        + (value % 100 == 0 ? "" : " " + cardinal(value % 100));
    }
    return cardinal(value / 1000) + " thousand"
      + (value % 1000 == 0 ? "" : " " + cardinal(value % 1000));
  }

  private static Map<String, String[]> letterNames() {
    Map<String, String[]> names = new LinkedHashMap<>();
    names.put("a", new String[] {"eɪ"});
    names.put("b", new String[] {"b", "iː"});
    names.put("c", new String[] {"s", "iː"});
    names.put("d", new String[] {"d", "iː"});
    names.put("e", new String[] {"iː"});
    names.put("f", new String[] {"ɛ", "f"});
    names.put("g", new String[] {"dʒ", "iː"});
    names.put("h", new String[] {"eɪ", "tʃ"});
    names.put("i", new String[] {"aɪ"});
    names.put("j", new String[] {"dʒ", "eɪ"});
    names.put("k", new String[] {"k", "eɪ"});
    names.put("l", new String[] {"ɛ", "l"});
    names.put("m", new String[] {"ɛ", "m"});
    names.put("n", new String[] {"ɛ", "n"});
    names.put("o", new String[] {"oʊ"});
    names.put("p", new String[] {"p", "iː"});
    names.put("q", new String[] {"k", "j", "uː"});
    names.put("r", new String[] {"ɑːɹ"});
    names.put("s", new String[] {"ɛ", "s"});
    names.put("t", new String[] {"t", "iː"});
    names.put("u", new String[] {"j", "uː"});
    names.put("v", new String[] {"v", "iː"});
    names.put("w", new String[] {"d", "ʌ", "b", "ə", "l", "j", "uː"});
    names.put("x", new String[] {"ɛ", "k", "s"});
    names.put("y", new String[] {"w", "aɪ"});
    names.put("z", new String[] {"z", "iː"});
    return Map.copyOf(names);
  }

  private static Map<String, String[]> fallback() {
    Map<String, String[]> map = new LinkedHashMap<>();
    map.put("r", new String[] {"ɹ"});
    map.put("a", new String[] {"æ"});
    map.put("e", new String[] {"eɪ"});
    map.put("o", new String[] {"oʊ"});
    map.put("u", new String[] {"uː"});
    map.put("əʊ", new String[] {"oʊ"});
    map.put("eə", new String[] {"ɛɹ"});
    return Map.copyOf(map);
  }

  @Override
  public Optional<TinyVocaUtterance> phonemize(final String text) {
    return TinyVocaText.assemble(sayNumbers(text), this::words, this.spec);
  }

  private List<List<Phone>> words(final String text) {
    List<List<Phone>> words = new ArrayList<>();
    for (String token : text.split("\\s+")) {
      for (String part : token.split("-")) {
        String word =
          part.toLowerCase(Locale.ROOT).replaceAll("^[^\\p{L}\\p{N}']+|[^\\p{L}\\p{N}']+$", "");
        if (word.isEmpty()) {
          continue;
        }
        words.add(this.pronounce(word));
      }
    }
    return words;
  }

  private List<Phone> pronounce(final String word) {
    String ipa = this.lexicon.get(word);
    if (ipa == null && word.endsWith("'s") && word.length() > 2) {
      List<Phone> stem = this.pronounce(word.substring(0, word.length() - 2));
      if (stem.isEmpty()) {
        return this.spell(word);
      }
      boolean sibilant = switch (stem.getLast().symbol()) {
        case "s", "z", "ʃ", "ʒ", "tʃ", "dʒ" -> true;
        default -> false;
      };
      if (sibilant) {
        stem.add(new Phone("ɪ", 0));
      }
      stem.add(new Phone("z", 0));
      return stem;
    }
    if (ipa == null) {
      return this.spell(word);
    }
    return this.phones(ipa);
  }

  private List<Phone> phones(final String ipa) {
    List<Phone> phones = new ArrayList<>();
    for (String token : ipa.split("\\s+")) {
      if (token.isBlank()) {
        continue;
      }
      int stress = token.indexOf('ˈ') >= 0 ? 1 : token.indexOf('ˌ') >= 0 ? 2 : 0;
      String body = token.replace("ˈ", "").replace("ˌ", "");
      List<String> symbols = this.expand(body);
      int carrier = 0;
      for (int i = 0; i < symbols.size(); i++) {
        if (this.spec.isVowel(this.spec.symbolId(symbols.get(i)))) {
          carrier = i;
          break;
        }
      }
      for (int i = 0; i < symbols.size(); i++) {
        phones.add(new Phone(symbols.get(i), i == carrier ? stress : 0));
      }
    }
    this.carryStressOntoVowels(phones);
    return phones;
  }

  private void carryStressOntoVowels(final List<Phone> phones) {
    for (int i = 0; i < phones.size(); i++) {
      Phone phone = phones.get(i);
      if (phone.stress() == 0 || this.isVowel(phone.symbol())) {
        continue;
      }
      for (int next = i + 1; next < phones.size(); next++) {
        if (!this.isVowel(phones.get(next).symbol())) {
          continue;
        }
        phones.set(i, new Phone(phone.symbol(), 0));
        if (phones.get(next).stress() == 0) {
          phones.set(next, new Phone(phones.get(next).symbol(), phone.stress()));
        }
        break;
      }
    }
  }

  private boolean isVowel(final String symbol) {
    return this.spec.isVowel(this.spec.symbolId(symbol));
  }

  private List<String> expand(final String token) {
    if (this.spec.symbolId(token) >= 0) {
      return List.of(token);
    }
    String[] mapped = FALLBACK.get(token);
    if (mapped != null) {
      return List.of(mapped);
    }
    List<String> greedy = new ArrayList<>();
    int cursor = 0;
    while (cursor < token.length()) {
      String matched = null;
      for (String symbol : this.symbolsByLength) {
        if (symbol.length() > 1 && token.startsWith(symbol, cursor)) {
          matched = symbol;
          break;
        }
      }
      if (matched == null && this.spec.symbolId(token.substring(cursor, cursor + 1)) >= 0) {
        matched = token.substring(cursor, cursor + 1);
      }
      if (matched == null) {
        return List.of("<unk>");
      }
      greedy.add(matched);
      cursor += matched.length();
    }
    return greedy;
  }

  private List<Phone> spell(final String word) {
    List<Phone> phones = new ArrayList<>();
    for (int i = 0; i < word.length(); i++) {
      String[] name = LETTERS.get(String.valueOf(word.charAt(i)));
      if (name == null) {
        continue;
      }
      for (int p = 0; p < name.length; p++) {
        phones.add(new Phone(name[p], p == 0 ? 1 : 0));
      }
    }
    if (phones.isEmpty()) {
      phones.add(new Phone("<unk>", 0));
    }
    return phones;
  }
}
