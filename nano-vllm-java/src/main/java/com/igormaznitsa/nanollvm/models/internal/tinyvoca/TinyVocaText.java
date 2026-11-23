package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Shared text cuts for tiny-voca: sentence breaks, punctuation pieces, and the
 * {@code <bos>} / {@code <eos>} wrapper.
 */
public final class TinyVocaText {

  private static final Pattern SPACE = Pattern.compile("[\\s\\uFEFF]+");
  private static final Pattern NOT_AN_END = Pattern.compile(
    "(?:\\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\\.g|i\\.e|a\\.m|p\\.m)|\\b[A-Z])\\.$",
    Pattern.CASE_INSENSITIVE);
  private static final String CLOSING = "\"'”’)]»";
  private static final Set<String> ABBREVIATIONS = Set.of(
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc");

  private TinyVocaText() {
  }

  public static String normalize(final String text) {
    String folded = Normalizer.normalize(requireNonNull(text, "text"), Normalizer.Form.NFKC);
    StringBuilder out = new StringBuilder(folded.length());
    for (int i = 0; i < folded.length(); i++) {
      out.append(switch (folded.charAt(i)) {
        case '‘', '’', '‚', '‛', '`', '´' -> '\'';
        case '“', '”', '„', '‟', '«', '»' -> '"';
        case '–', '―' -> '—';
        case '[', '{' -> '(';
        case ']', '}' -> ')';
        default -> folded.charAt(i);
      });
    }
    String quotes = out.toString().replaceAll("\\.{3,}", "…").replaceAll("-{2,}|\\s-\\s", "—");
    return SPACE.matcher(quotes).replaceAll(" ").strip();
  }

  public static List<String> sentences(final String text) {
    String flat = SPACE.matcher(text).replaceAll(" ").strip();
    List<String> parts = new ArrayList<>();
    StringBuilder current = new StringBuilder();
    int index = 0;
    while (index < flat.length()) {
      Split split = split(flat, index);
      if (split == null) {
        current.append(flat.charAt(index));
        index++;
        continue;
      }
      String piece = current + flat.substring(index, split.pieceEnd());
      if (NOT_AN_END.matcher(piece).find()) {
        current.append(flat.charAt(index));
        index++;
        continue;
      }
      parts.add(piece.strip());
      current.setLength(0);
      index = split.nextStart();
    }
    if (!current.isEmpty()) {
      parts.add(current.toString().strip());
    }
    return parts.stream().filter(part -> part.chars().anyMatch(Character::isLetterOrDigit))
      .toList();
  }

  public static String finished(final String text) {
    String body = SPACE.matcher(text).replaceAll(" ").strip();
    if (body.isEmpty()) {
      return text;
    }
    int lastSpace = body.lastIndexOf(' ');
    String last = lastSpace < 0 ? body : body.substring(lastSpace + 1);
    boolean marked;
    if (last.chars().anyMatch(Character::isLetterOrDigit)) {
      int end = last.length();
      while (end > 0 && CLOSING.indexOf(last.charAt(end - 1)) >= 0) {
        end--;
      }
      char mark = end == 0 ? 0 : last.charAt(end - 1);
      marked = ".!?…,;:".indexOf(mark) >= 0;
    } else {
      marked = last.chars().anyMatch(ch -> ".!?…".indexOf(ch) >= 0)
        || ",".equals(last) || ";".equals(last) || ":".equals(last);
    }
    return marked ? text : body + ".";
  }

  public static List<String> shorter(final String text) {
    List<String> clauses = new ArrayList<>();
    StringBuilder current = new StringBuilder();
    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);
      current.append(ch);
      if (",;:—".indexOf(ch) >= 0 && i + 1 < text.length() &&
        Character.isWhitespace(text.charAt(i + 1))) {
        clauses.add(current.toString().strip());
        current.setLength(0);
      }
    }
    if (!current.isEmpty()) {
      clauses.add(current.toString().strip());
    }
    clauses.removeIf(String::isBlank);
    if (clauses.size() > 1) {
      return clauses;
    }
    String[] words = text.strip().split(" ");
    if (words.length < 2) {
      return List.of();
    }
    int half = (words.length + 1) / 2;
    return List.of(
      String.join(" ", java.util.Arrays.copyOf(words, half)),
      String.join(" ", java.util.Arrays.copyOfRange(words, half, words.length)));
  }

  public static List<Piece> chunks(final String text) {
    List<Piece> pieces = new ArrayList<>();
    int start = 0;
    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);
      if (!isPunctuation(ch)) {
        continue;
      }
      char previous = i == 0 ? 0 : text.charAt(i - 1);
      char following = i + 1 < text.length() ? text.charAt(i + 1) : 0;
      if (ch == '\'' && Character.isLetterOrDigit(previous) &&
        Character.isLetterOrDigit(following)) {
        continue;
      }
      if ((ch == '.' || ch == ',' || ch == ':') && Character.isDigit(previous) &&
        Character.isDigit(following)) {
        continue;
      }
      if (ch == '.') {
        String word = trailingLetters(text, i);
        if (word != null && ABBREVIATIONS.contains(word.toLowerCase(Locale.ROOT))) {
          continue;
        }
        if (Character.isLetter(previous) && Character.isLetter(following)) {
          continue;
        }
      }
      String between = text.substring(start, i).strip();
      if (!between.isEmpty()) {
        pieces.add(new Piece(false, between));
      }
      pieces.add(new Piece(true, String.valueOf(ch)));
      start = i + 1;
    }
    String tail = text.substring(start).strip();
    if (!tail.isEmpty()) {
      pieces.add(new Piece(false, tail));
    }
    return pieces;
  }

  public static Optional<TinyVocaUtterance> assemble(
    final String text,
    final WordReader reader,
    final TinyVocaSpec spec
  ) {
    requireNonNull(reader, "reader");
    requireNonNull(spec, "spec");
    List<String> symbols = new ArrayList<>();
    List<Integer> stress = new ArrayList<>();
    symbols.add("<bos>");
    stress.add(0);
    for (Piece piece : chunks(normalize(text))) {
      if (piece.punctuation()) {
        if (!symbols.getLast().equals(piece.text())) {
          symbols.add(piece.text());
          stress.add(0);
        }
        continue;
      }
      for (List<Phone> word : reader.read(piece.text())) {
        if (word.isEmpty()) {
          continue;
        }
        String last = symbols.getLast();
        if (spec.symbolId(last) >= 0 && (spec.isVowel(spec.symbolId(last))
          || !spec.isPause(spec.symbolId(last)) && !"<bos>".equals(last))) {
          symbols.add("<sp>");
          stress.add(0);
        }
        for (Phone phone : word) {
          symbols.add(phone.symbol());
          stress.add(phone.stress());
        }
      }
    }
    symbols.add("<eos>");
    stress.add(0);
    boolean spoken = false;
    int[] ids = new int[symbols.size()];
    int[] levels = new int[symbols.size()];
    int unk = spec.symbolId("<unk>");
    for (int i = 0; i < symbols.size(); i++) {
      int id = spec.symbolId(symbols.get(i));
      if (id < 0) {
        id = unk;
      }
      ids[i] = id;
      levels[i] = stress.get(i);
      if (!spec.isPause(id)) {
        spoken = true;
      }
    }
    if (!spoken) {
      return Optional.empty();
    }
    return Optional.of(new TinyVocaUtterance(ids, levels));
  }

  private static boolean isPunctuation(final char ch) {
    return ",.?!;:\"'()—…".indexOf(ch) >= 0;
  }

  private static String trailingLetters(final String text, final int dot) {
    int start = dot;
    while (start > 0 && Character.isLetter(text.charAt(start - 1))) {
      start--;
    }
    return start == dot ? null : text.substring(start, dot);
  }

  private static Split split(final String text, final int index) {
    char ch = text.charAt(index);
    if (".!?…".indexOf(ch) < 0) {
      return null;
    }
    int cursor = index + 1;
    if (cursor < text.length() && "\"')]".indexOf(text.charAt(cursor)) >= 0) {
      cursor++;
    }
    int pieceEnd = cursor;
    if (cursor >= text.length() || !Character.isWhitespace(text.charAt(cursor))) {
      return null;
    }
    while (cursor < text.length() && Character.isWhitespace(text.charAt(cursor))) {
      cursor++;
    }
    if (cursor >= text.length()) {
      return null;
    }
    char next = text.charAt(cursor);
    if ("\"'([".indexOf(next) >= 0 && cursor + 1 < text.length()) {
      next = text.charAt(cursor + 1);
    }
    if (Character.isUpperCase(next) || Character.isDigit(next)) {
      return new Split(pieceEnd, cursor);
    }
    return null;
  }

  @FunctionalInterface
  public interface WordReader {
    List<List<Phone>> read(String text);
  }

  /**
   * One lexicon phone and its stress, {@code 0} none, {@code 1} primary, {@code 2} secondary.
   *
   * @param symbol inventory token
   * @param stress stress feature
   */
  public record Phone(String symbol, int stress) {
  }

  /**
   * A punctuation mark or a stretch of words.
   *
   * @param punctuation {@code true} when {@code text} is one mark
   * @param text        the mark or the word stretch
   */
  public record Piece(boolean punctuation, String text) {
  }

  private record Split(int pieceEnd, int nextStart) {
  }
}
