package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.models.internal.tinyvoca.TinyVocaText.Phone;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/**
 * Russian front end for the same acoustic graph: letter rules, {@code +} or {@code ё} stress,
 * and an optional {@code stress.txt} / {@code stress.txt.gz} beside the checkpoint.
 * A missing dictionary stresses the last vowel.
 */
public final class RussianVocaFront implements TinyVocaFront {

  private static final String VOWELS = "аеёиоуыэюя";
  private static final String IOTATED = "еёюя";
  private static final String ALWAYS_HARD = "жшц";
  private static final String ALWAYS_SOFT = "чщ";
  private static final Locale RU = Locale.forLanguageTag("ru");
  private static final Map<Character, String> HARD = hardConsonants();
  private static final Map<Character, String> SOFT = softConsonants();
  private static final Map<String, String> VOICE = voicingPairs();
  private static final Set<String> PROCLITICS = Set.of(
    "без", "безо", "в", "во", "для", "до", "за", "из", "изо", "к", "ко",
    "на", "над", "надо", "о", "об", "обо", "от", "ото", "по", "под", "подо",
    "пред", "предо", "при", "про", "с", "со", "у", "через", "сквозь", "между",
    "перед", "передо", "не", "ни");
  private static final Set<String> ENCLITICS = Set.of("бы", "б", "же", "ж", "ли", "ль");
  private static final String[] SMALL = {
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"
  };
  private static final String[] TENS = {
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
    "семьдесят", "восемьдесят", "девяносто"
  };
  private static final String[] HUNDREDS = {
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
    "семьсот", "восемьсот", "девятьсот"
  };
  private final TinyVocaSpec spec;
  private final Map<String, Integer> dictionary;

  public RussianVocaFront(final TinyVocaSpec spec, final Path modelDir) {
    this.spec = requireNonNull(spec, "spec");
    this.dictionary = modelDir == null ? Map.of() : loadStress(modelDir);
  }

  private static int vowelsBefore(final String text, final int end) {
    int count = 0;
    for (int i = 0; i < end && i < text.length(); i++) {
      if (VOWELS.indexOf(text.charAt(i)) >= 0) {
        count++;
      }
    }
    return count;
  }

  private static Map<String, Integer> loadStress(final Path modelDir) {
    Path plain = firstExisting(
      modelDir.resolve("phonemizer").resolve("stress.txt"),
      modelDir.resolve("stress.txt"));
    Path gzip = firstExisting(
      modelDir.resolve("phonemizer").resolve("stress.txt.gz"),
      modelDir.resolve("stress.txt.gz"));
    try {
      if (plain != null) {
        return readStress(Files.newBufferedReader(plain, UTF_8));
      }
      if (gzip != null) {
        return readStress(new BufferedReader(
          new InputStreamReader(new GZIPInputStream(Files.newInputStream(gzip)), UTF_8)));
      }
    } catch (IOException ignored) {
      return Map.of();
    }
    return Map.of();
  }

  private static Map<String, Integer> readStress(final BufferedReader reader) throws IOException {
    Map<String, Integer> words = new LinkedHashMap<>();
    try (reader) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank() || line.startsWith("#")) {
          continue;
        }
        String[] parts = line.split("\\t");
        if (parts.length < 2) {
          continue;
        }
        String key = parts[0].toLowerCase(RU).replace("ё", "е");
        String value = parts[1].strip();
        if (value.chars().allMatch(Character::isDigit)) {
          words.put(key, Integer.parseInt(value));
          continue;
        }
        int mark = value.indexOf('+');
        if (mark < 0) {
          mark = value.indexOf('\u0301');
        }
        if (mark >= 0) {
          words.put(key, vowelsBefore(value.replace("+", ""), mark));
        }
      }
    }
    return Map.copyOf(words);
  }

  private static Path firstExisting(final Path first, final Path second) {
    if (Files.isRegularFile(first)) {
      return first;
    }
    if (Files.isRegularFile(second)) {
      return second;
    }
    return null;
  }

  private static String sayNumbers(final String text) {
    StringBuilder out = new StringBuilder();
    int index = 0;
    while (index < text.length()) {
      if (!Character.isDigit(text.charAt(index))) {
        out.append(text.charAt(index));
        index++;
        continue;
      }
      int end = index;
      while (end < text.length() && Character.isDigit(text.charAt(end))) {
        end++;
      }
      String digits = text.substring(index, end);
      long value = Long.parseLong(digits.length() > 6 ? digits.substring(0, 6) : digits);
      out.append(value < 1000 ? cardinal((int) value) : digits);
      index = end;
    }
    return out.toString();
  }

  private static String cardinal(final int value) {
    if (value < 20) {
      return SMALL[value];
    }
    if (value < 100) {
      return TENS[value / 10] + (value % 10 == 0 ? "" : " " + SMALL[value % 10]);
    }
    int hundreds = value / 100;
    int rest = value % 100;
    return HUNDREDS[hundreds] + (rest == 0 ? "" : " " + cardinal(rest));
  }

  private static List<List<String>> groupClitics(final List<String> tokens) {
    List<List<String>> groups = new ArrayList<>();
    List<String> pending = new ArrayList<>();
    for (String word : tokens) {
      String key = lettersOf(word);
      boolean marked = word.indexOf('+') >= 0 || key.indexOf('ё') >= 0;
      if (PROCLITICS.contains(key) && !marked) {
        pending.add(word);
        continue;
      }
      if (ENCLITICS.contains(key) && !groups.isEmpty() && pending.isEmpty()) {
        groups.getLast().add(word);
        continue;
      }
      List<String> group = new ArrayList<>(pending);
      pending.clear();
      group.add(word);
      groups.add(group);
    }
    if (!pending.isEmpty()) {
      if (groups.isEmpty()) {
        groups.add(pending);
      } else {
        groups.getLast().addAll(pending);
      }
    }
    return groups;
  }

  private static String lettersOf(final String marked) {
    return marked.replace("+", "").replace("\u0301", "");
  }

  private static Map<Character, String> hardConsonants() {
    Map<Character, String> map = new LinkedHashMap<>();
    map.put('б', "b");
    map.put('в', "v");
    map.put('г', "ɡ");
    map.put('д', "d");
    map.put('з', "z");
    map.put('к', "k");
    map.put('л', "l");
    map.put('м', "m");
    map.put('н', "n");
    map.put('п', "p");
    map.put('р', "r");
    map.put('с', "s");
    map.put('т', "t");
    map.put('ф', "f");
    map.put('х', "x");
    return Map.copyOf(map);
  }

  private static Map<Character, String> softConsonants() {
    Map<Character, String> map = new LinkedHashMap<>();
    hardConsonants().forEach((letter, phone) -> map.put(letter, phone + "ʲ"));
    return Map.copyOf(map);
  }

  private static Map<String, String> voicingPairs() {
    Map<String, String> map = new LinkedHashMap<>();
    String[] voiced = {"b", "bʲ", "v", "vʲ", "ɡ", "ɡʲ", "d", "dʲ", "z", "zʲ", "ʐ"};
    String[] quiet = {"p", "pʲ", "f", "fʲ", "k", "kʲ", "t", "tʲ", "s", "sʲ", "ʂ"};
    for (int i = 0; i < voiced.length; i++) {
      map.put(voiced[i], quiet[i]);
    }
    return Map.copyOf(map);
  }

  @Override
  public Optional<TinyVocaUtterance> phonemize(final String text) {
    return TinyVocaText.assemble(sayNumbers(text), this::words, this.spec);
  }

  private List<List<Phone>> words(final String text) {
    List<String> tokens = new ArrayList<>();
    for (String token : text.toLowerCase(RU).split("\\s+")) {
      String word = token.replaceAll("^[^\\p{L}+]+|[^\\p{L}+]+$", "");
      if (word.isEmpty() || word.indexOf('+') == word.length() - 1 || !this.hasSpeechLetter(word)) {
        continue;
      }
      tokens.add(word);
    }
    List<List<Phone>> spoken = new ArrayList<>();
    for (List<String> group : groupClitics(tokens)) {
      spoken.add(this.pronounceGroup(group));
    }
    return spoken;
  }

  private List<Phone> pronounceGroup(final List<String> words) {
    int host = this.hostIndex(words);
    String hostLetters = lettersOf(words.get(host));
    int hostStress = this.stressIndex(words.get(host), hostLetters);
    List<Segment> segments = new ArrayList<>();
    int vowels = 0;
    int hostVowel = 0;
    int stressedAt = -1;
    for (int wordIndex = 0; wordIndex < words.size(); wordIndex++) {
      String letters = lettersOf(words.get(wordIndex));
      for (int i = 0; i < letters.length(); i++) {
        char ch = letters.charAt(i);
        if (VOWELS.indexOf(ch) >= 0) {
          boolean stressed = wordIndex == host && hostVowel == hostStress;
          this.addVowel(segments, letters, i, ch, stressed);
          if (stressed) {
            stressedAt = vowels;
          }
          vowels++;
          if (wordIndex == host) {
            hostVowel++;
          }
          continue;
        }
        if (ch == 'ь' || ch == 'ъ') {
          continue;
        }
        String consonant = this.consonant(letters, i, ch);
        if (consonant != null) {
          segments.add(new Segment(consonant, ch, false, false));
        }
      }
    }
    this.markPretonic(segments, stressedAt);
    this.assimilate(segments);
    List<Phone> spoken = new ArrayList<>(segments.size());
    for (Segment segment : segments) {
      spoken.add(new Phone(this.resolve(segment.phone()), segment.stressed() ? 1 : 0));
    }
    if (spoken.isEmpty()) {
      spoken.add(new Phone("<unk>", 0));
    }
    return spoken;
  }

  private void addVowel(
    final List<Segment> segments,
    final String letters,
    final int index,
    final char vowel,
    final boolean stressed
  ) {
    boolean iotated = IOTATED.indexOf(vowel) >= 0;
    boolean needsGlide = iotated && (index == 0 || this.breakBefore(letters.charAt(index - 1)));
    if (needsGlide) {
      segments.add(new Segment("j", 'й', false, false));
    }
    segments.add(new Segment(this.vowelQuality(vowel, stressed, false), vowel, true, stressed));
  }

  private void markPretonic(final List<Segment> segments, final int stressedAt) {
    if (stressedAt <= 0) {
      return;
    }
    int seen = 0;
    for (int i = 0; i < segments.size(); i++) {
      Segment segment = segments.get(i);
      if (!segment.vowel()) {
        continue;
      }
      if (seen == stressedAt - 1 && "ə".equals(segment.phone())) {
        segments.set(i, segment.withPhone("ɐ"));
        return;
      }
      seen++;
    }
  }

  private String vowelQuality(final char vowel, final boolean stressed, final boolean pretonic) {
    return switch (vowel) {
      case 'а', 'я' -> stressed ? "a" : pretonic ? "ɐ" : "ə";
      case 'о', 'ё' -> stressed ? "o" : pretonic ? "ɐ" : "ə";
      case 'э', 'е' -> stressed ? "e" : "i";
      case 'ы' -> "ɨ";
      case 'и' -> "i";
      case 'у', 'ю' -> "u";
      default -> "<unk>";
    };
  }

  private String consonant(final String letters, final int index, final char ch) {
    if (ch == 'й') {
      return "j";
    }
    if (ch == 'щ') {
      return "ɕ";
    }
    if (ch == 'ч') {
      return "tɕ";
    }
    if (ch == 'ц') {
      return "ts";
    }
    if (ch == 'ш') {
      return "ʂ";
    }
    if (ch == 'ж') {
      return "ʐ";
    }
    boolean soft = ALWAYS_SOFT.indexOf(ch) >= 0 ||
      (ALWAYS_HARD.indexOf(ch) < 0 && this.softAfter(letters, index));
    Map<Character, String> table = soft ? SOFT : HARD;
    return table.get(ch);
  }

  private boolean softAfter(final String letters, final int index) {
    int next = index + 1;
    if (next >= letters.length()) {
      return false;
    }
    char follow = letters.charAt(next);
    if (follow == 'ь') {
      return true;
    }
    if (follow == 'ъ') {
      return false;
    }
    return "еиёюя".indexOf(follow) >= 0;
  }

  private boolean breakBefore(final char previous) {
    return previous == 'ь' || previous == 'ъ' || VOWELS.indexOf(previous) >= 0;
  }

  private void assimilate(final List<Segment> segments) {
    String state = "voiceless";
    for (int i = segments.size() - 1; i >= 0; i--) {
      Segment segment = segments.get(i);
      if (!this.isObstruent(segment.phone())) {
        state = null;
        continue;
      }
      String phone = segment.phone();
      if ("voiceless".equals(state)) {
        phone = this.devoice(phone);
      } else if ("voiced".equals(state)) {
        phone = this.voice(phone);
      }
      segments.set(i, segment.withPhone(phone));
      if (segment.letter() != 'в') {
        state = this.isVoiced(phone) ? "voiced" : "voiceless";
      }
    }
  }

  private boolean isObstruent(final String phone) {
    return VOICE.containsKey(phone) ||
      this.devoice(phone) != null && VOICE.containsKey(this.voice(phone));
  }

  private boolean isVoiced(final String phone) {
    return VOICE.containsKey(phone);
  }

  private String voice(final String phone) {
    for (var entry : VOICE.entrySet()) {
      if (entry.getValue().equals(phone) || entry.getKey().equals(phone)) {
        return entry.getKey();
      }
    }
    return phone;
  }

  private String devoice(final String phone) {
    return VOICE.getOrDefault(phone, phone);
  }

  private String resolve(final String phone) {
    if (this.spec.symbolId(phone) >= 0) {
      return phone;
    }
    if ("g".equals(phone) && this.spec.symbolId("ɡ") >= 0) {
      return "ɡ";
    }
    return "<unk>";
  }

  private int stressIndex(final String marked, final String letters) {
    int plus = marked.indexOf('+');
    if (plus >= 0) {
      return vowelsBefore(marked, plus);
    }
    int acute = marked.indexOf('\u0301');
    if (acute > 0) {
      return Math.max(0, vowelsBefore(marked, acute) - 1);
    }
    int yo = letters.indexOf('ё');
    if (yo >= 0) {
      return vowelsBefore(letters, yo + 1);
    }
    String key = letters.replace("ё", "е");
    Integer listed = this.dictionary.get(key);
    if (listed != null) {
      return listed;
    }
    int count = 0;
    for (int i = 0; i < letters.length(); i++) {
      if (VOWELS.indexOf(letters.charAt(i)) >= 0) {
        count++;
      }
    }
    return count - 1;
  }

  private int hostIndex(final List<String> words) {
    for (int i = 0; i < words.size(); i++) {
      String key = lettersOf(words.get(i));
      if (!PROCLITICS.contains(key) && !ENCLITICS.contains(key)) {
        return i;
      }
    }
    return words.size() - 1;
  }

  private boolean hasSpeechLetter(final String word) {
    for (int i = 0; i < word.length(); i++) {
      char ch = word.charAt(i);
      if (VOWELS.indexOf(ch) >= 0 || ch == '+' || ch == 'й' || HARD.containsKey(ch)
        || "жшцчщ".indexOf(ch) >= 0) {
        return true;
      }
    }
    return false;
  }

  private record Segment(String phone, char letter, boolean vowel, boolean stressed) {
    private Segment withPhone(final String phone) {
      return new Segment(phone, this.letter, this.vowel, this.stressed);
    }
  }
}
