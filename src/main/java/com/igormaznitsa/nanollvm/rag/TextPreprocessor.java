package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Document cleanup for {@link RagFactory}: Markdown noise removal, section titles,
 * sentence-sized passages. Language-agnostic (Unicode letters, structural Markdown only).
 */
final class TextPreprocessor {

  private static final Pattern HEADING_LINE = Pattern.compile("^#{1,6}\\s+(.+?)\\s*$");
  private static final Pattern LINK = Pattern.compile("\\[([^\\]]+)]\\([^)]*\\)");
  private static final Pattern IMAGE = Pattern.compile("!\\[[^\\]]*]\\([^)]*\\)");
  private static final Pattern CODE_FENCE = Pattern.compile("(?s)```.*?```");
  private static final Pattern INLINE_CODE = Pattern.compile("`([^`]+)`");
  private static final Pattern BULLET = Pattern.compile("^\\s*[-*+]\\s+");
  private static final Pattern SENTENCE_END =
      Pattern.compile("(?<=[.!?。！？])\\s+(?=[\\p{L}\\p{N}\"'(«])");

  private TextPreprocessor() {
  }

  /**
   * Turns raw document text into model-ready passages (optional {@code Section — sentence}).
   */
  static List<String> passages(String raw) {
    if (raw == null || raw.isBlank()) {
      return List.of();
    }
    String text =
        CODE_FENCE.matcher(raw.replace("\r\n", "\n").replace('\r', '\n')).replaceAll("\n");
    String section = "";
    StringBuilder paragraph = new StringBuilder();
    List<String> out = new ArrayList<>();

    for (String line : text.split("\n", -1)) {
      Matcher heading = HEADING_LINE.matcher(line.strip());
      if (heading.matches()) {
        flushParagraph(paragraph, section, out);
        section = cleanInline(heading.group(1)).strip();
        continue;
      }
      if (line.isBlank()) {
        flushParagraph(paragraph, section, out);
        continue;
      }
      String cleaned = cleanInline(BULLET.matcher(line).replaceFirst("")).strip();
      if (cleaned.isEmpty()) {
        continue;
      }
      if (!paragraph.isEmpty()) {
        paragraph.append(' ');
      }
      paragraph.append(cleaned);
    }
    flushParagraph(paragraph, section, out);
    return List.copyOf(out);
  }

  static String normalize(String raw) {
    return String.join("\n\n", passages(raw));
  }

  static List<String> units(String normalized) {
    requireNonNull(normalized, "normalized");
    if (normalized.isBlank()) {
      return List.of();
    }
    return passages(normalized);
  }

  private static void flushParagraph(StringBuilder paragraph, String section, List<String> out) {
    if (paragraph.isEmpty()) {
      return;
    }
    String body = paragraph.toString().replaceAll(" +", " ").strip();
    paragraph.setLength(0);
    if (body.isEmpty()) {
      return;
    }
    for (String sentence : SENTENCE_END.split(body)) {
      String s = sentence.strip();
      if (s.isEmpty()) {
        continue;
      }
      out.add(section.isBlank() ? s : section + " — " + s);
    }
  }

  private static String cleanInline(String line) {
    String text = IMAGE.matcher(line).replaceAll("");
    text = LINK.matcher(text).replaceAll("$1");
    text = INLINE_CODE.matcher(text).replaceAll("$1");
    text = text.replace('\t', ' ').replaceAll("[ ]{2,}", " ");
    return text;
  }
}
