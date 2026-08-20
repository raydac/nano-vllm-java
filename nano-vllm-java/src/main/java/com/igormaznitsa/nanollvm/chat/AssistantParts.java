package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Package-private split of decoded assistant text into thinking / answer / {@code thinkOpen}.
 * {@link ChatReply} is the public facade.
 */
record AssistantParts(String thinking, String answer, boolean thinkOpen) {

  private static final Pattern LINE_BREAK = Pattern.compile("\\R");
  private static final Pattern LEADING_ASSISTANT = Pattern.compile("(?i)^\\s*assistant\\s*:?\\s*");
  private static final Pattern STATED_ANSWER = Pattern.compile(
    "(?i)(?:answer(?:\\s+should)?\\s+be|returns?|result(?:\\s+is)?)\\s*[:\"']?\\s*(.+)$");

  /**
   * {@link #parse(String, ThinkTags, ChatSpecials)} with library default markers.
   */
  public static AssistantParts parse(final String raw) {
    return parse(raw, ThinkTags.DEFAULT, ChatSpecials.DEFAULT);
  }

  /** {@link #parse(String, ThinkTags, ChatSpecials)} with {@link ChatSpecials#DEFAULT}. */
  public static AssistantParts parse(final String raw, final ThinkTags tags) {
    return parse(raw, tags, ChatSpecials.DEFAULT);
  }

  /**
   * Splits decoded assistant text into thinking vs answer using {@code tags} / {@code specials}.
   *
   * @param raw      decoded tokens; {@code null} / blank → empty channels
   * @param tags     scratchpad pair
   * @param specials chat markup (think tags are merged in)
   * @return parsed parts
   */
  public static AssistantParts parse(
    final String raw,
    final ThinkTags tags,
    final ChatSpecials specials
  ) {
    requireNonNull(tags, "tags");
    requireNonNull(specials, "specials");
    if (raw == null || raw.isBlank()) {
      return new AssistantParts("", "", false);
    }

    List<String> markup = specials.searchMarkers(tags);
    String text = holdIncompleteMarkupSuffix(raw, markup);
    String open = tags.open();
    String close = tags.close();
    int startThink = text.indexOf(open);
    int endThink = startThink >= 0
      ? text.indexOf(close, startThink + open.length())
      : text.indexOf(close);

    String thinking;
    boolean thinkOpen;
    String after;
    if (startThink >= 0 && endThink > startThink) {
      thinking = text.substring(startThink + open.length(), endThink);
      Remainder rest = splitRemainder(text.substring(endThink + close.length()), tags, markup);
      thinking = joinThink(thinking, rest.thinking());
      after = rest.answer();
      thinkOpen = rest.open();
    } else if (startThink >= 0) {
      thinking = text.substring(startThink + open.length());
      after = "";
      thinkOpen = true;
    } else if (endThink >= 0) {
      Remainder rest = splitRemainder(text.substring(endThink + close.length()), tags, markup);
      thinking = rest.thinking();
      after = rest.answer();
      thinkOpen = rest.open();
    } else {
      thinking = "";
      after = text;
      thinkOpen = false;
    }

    return new AssistantParts(stripChatMarkup(thinking, tags, specials),
      sanitizeAnswer(after, markup),
      thinkOpen);
  }

  /**
   * Recursively splits text after a closed think block in case another think pair follows.
   */
  private static Remainder splitRemainder(
    final String after,
    final ThinkTags tags,
    final List<String> markup
  ) {
    final String text = holdIncompleteMarkupSuffix(after, markup);
    String open = tags.open();
    int start = text.indexOf(open);
    if (start < 0) {
      return new Remainder("", text, false);
    }

    String before = text.substring(0, start);
    String fromThink = text.substring(start + open.length());
    String close = tags.close();
    int end = fromThink.indexOf(close);
    if (end < 0) {
      return new Remainder(fromThink, before, true);
    }

    String inner = fromThink.substring(0, end);
    Remainder nested = splitRemainder(fromThink.substring(end + close.length()), tags, markup);
    return new Remainder(
      joinThink(inner, nested.thinking()),
      joinAnswer(before, nested.answer()),
      nested.open());
  }

  /**
   * Joins two think bodies with a newline; blank sides are dropped.
   */
  private static String joinThink(final String first, final String second) {
    String a = first == null ? "" : first.strip();
    String b = second == null ? "" : second.strip();
    if (a.isEmpty()) {
      return b;
    }
    if (b.isEmpty()) {
      return a;
    }
    return a + "\n" + b;
  }

  /**
   * Joins two answer fragments, preserving inner whitespace on the non-blank sides.
   */
  private static String joinAnswer(final String first, final String second) {
    String a = first == null ? "" : first;
    String b = second == null ? "" : second;
    if (a.isBlank()) {
      return b;
    }
    if (b.isBlank()) {
      return a;
    }
    return a.stripTrailing() + "\n" + b.stripLeading();
  }

  /**
   * Truncates the answer at the first chat special and strips a leading {@code assistant:} prefix.
   */
  private static String sanitizeAnswer(final String text, final List<String> markup) {
    String cleaned = holdIncompleteMarkupSuffix(text, markup);
    for (String marker : markup) {
      int at = cleaned.indexOf(marker);
      if (at >= 0) {
        cleaned = cleaned.substring(0, at);
      }
    }
    return finishSanitize(cleaned);
  }

  /** {@link #stripChatMarkup(String, ThinkTags, ChatSpecials)} with library default markers. */
  public static String stripChatMarkup(final String text) {
    return stripChatMarkup(text, ThinkTags.DEFAULT, ChatSpecials.DEFAULT);
  }

  /** {@link #stripChatMarkup(String, ThinkTags, ChatSpecials)} with {@link ChatSpecials#DEFAULT}. */
  public static String stripChatMarkup(final String text, final ThinkTags tags) {
    return stripChatMarkup(text, tags, ChatSpecials.DEFAULT);
  }

  /**
   * Removes think tags and chat specials, then a leading {@code assistant:} prefix.
   *
   * @param text     decoded fragment; {@code null} / empty → {@code ""}
   * @param tags     scratchpad pair
   * @param specials chat markup (think tags are merged in)
   * @return stripped text
   */
  public static String stripChatMarkup(
    final String text,
    final ThinkTags tags,
    final ChatSpecials specials
  ) {
    requireNonNull(tags, "tags");
    requireNonNull(specials, "specials");
    if (text == null || text.isEmpty()) {
      return "";
    }
    List<String> markup = specials.searchMarkers(tags);
    String s = holdIncompleteMarkupSuffix(text, markup);
    for (String marker : markup) {
      s = s.replace(marker, "");
    }
    return finishSanitize(s);
  }

  /**
   * Strips a leading {@code assistant:} label and surrounding whitespace.
   */
  private static String finishSanitize(final String text) {
    return LEADING_ASSISTANT.matcher(text).replaceFirst("").strip();
  }

  /** {@link #cleanAssistantText(String, ThinkTags, ChatSpecials)} with library default markers. */
  public static String cleanAssistantText(final String raw) {
    return cleanAssistantText(raw, ThinkTags.DEFAULT, ChatSpecials.DEFAULT);
  }

  /** {@link #cleanAssistantText(String, ThinkTags, ChatSpecials)} with {@link ChatSpecials#DEFAULT}. */
  public static String cleanAssistantText(final String raw, final ThinkTags tags) {
    return cleanAssistantText(raw, tags, ChatSpecials.DEFAULT);
  }

  /**
   * Visible answer after parse; falls back to {@link #salvageFromThinking(String)} when empty.
   *
   * @param raw      decoded tokens
   * @param tags     scratchpad pair
   * @param specials chat markup
   * @return visible answer (possibly salvaged)
   */
  public static String cleanAssistantText(
    final String raw,
    final ThinkTags tags,
    final ChatSpecials specials
  ) {
    AssistantParts parts = parse(raw, tags, specials);
    String answer = parts.answer();
    return answer.isEmpty() ? salvageFromThinking(parts.thinking()) : answer;
  }

  /** Same as {@link #cleanAssistantText(String)} (CLI stream display). */
  public static String streamDisplayText(final String raw) {
    return streamDisplayText(raw, ThinkTags.DEFAULT, ChatSpecials.DEFAULT);
  }

  /** Same as {@link #cleanAssistantText(String, ThinkTags)}. */
  public static String streamDisplayText(final String raw, final ThinkTags tags) {
    return streamDisplayText(raw, tags, ChatSpecials.DEFAULT);
  }

  /** Same as {@link #cleanAssistantText(String, ThinkTags, ChatSpecials)}. */
  public static String streamDisplayText(
    final String raw,
    final ThinkTags tags,
    final ChatSpecials specials
  ) {
    return cleanAssistantText(raw, tags, specials);
  }

  /**
   * Last-resort visible reply from a thinking block when the parsed answer is empty.
   */
  public static String salvageFromThinking(final String thinking) {
    if (thinking == null || thinking.isBlank()) {
      return "";
    }

    String[] lines = LINE_BREAK.split(thinking.strip(), -1);
    for (int i = lines.length - 1; i >= 0; i--) {
      String line = lines[i].strip();
      if (line.isEmpty() || line.startsWith("+")) {
        continue;
      }
      Matcher stated = STATED_ANSWER.matcher(line);
      if (stated.find()) {
        String extracted = stated.group(1).strip().replaceAll("[.\"']+$", "");
        if (!extracted.isEmpty()) {
          return extracted;
        }
      }
      if (isBriefAnswerLine(line)) {
        return line;
      }
    }

    for (int i = lines.length - 1; i >= 0; i--) {
      String line = lines[i].strip();
      if (line.isEmpty() || line.startsWith("+")) {
        continue;
      }
      return line.length() > 200 ? line.substring(0, 200).strip() + "…" : line;
    }

    String one = thinking.replace('\n', ' ').strip();
    return one.length() > 200 ? one.substring(0, 200).strip() + "…" : one;
  }

  /**
   * Short last line that does not look like chain-of-thought preamble.
   */
  private static boolean isBriefAnswerLine(final String line) {
    return line.length() <= 32 && !looksLikeReasoning(line);
  }

  /**
   * {@code true} for common reasoning openers ({@code okay}, {@code let me}, {@code the user}, …).
   */
  private static boolean looksLikeReasoning(final String line) {
    String lower = line.toLowerCase(java.util.Locale.ROOT);
    return lower.startsWith("okay")
      || lower.startsWith("wait")
      || lower.startsWith("let me")
      || lower.startsWith("i think")
      || lower.startsWith("i need")
      || lower.contains("the user")
      || lower.contains("user wants")
      || lower.contains("user said");
  }

  /**
   * Drop a trailing incomplete chat/think marker so streamed decode does not leak
   * fragments like {@code <think} into the answer channel.
   */
  static String holdIncompleteMarkupSuffix(final String text) {
    return holdIncompleteMarkupSuffix(text, ChatSpecials.DEFAULT.searchMarkers(ThinkTags.DEFAULT));
  }

  /**
   * Same as {@link #holdIncompleteMarkupSuffix(String)} against an explicit marker list.
   */
  static String holdIncompleteMarkupSuffix(final String text, final List<String> markers) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    int longest = markers.stream().mapToInt(String::length).max().orElse(0);
    int minStart = Math.max(0, text.length() - longest + 1);
    for (int i = minStart; i < text.length(); i++) {
      String suffix = text.substring(i);
      if (isStrictPrefixOfMarkup(suffix, markers)) {
        return text.substring(0, i);
      }
    }
    return text;
  }

  /**
   * {@code true} when {@code suffix} is a proper prefix of some marker (not the full marker).
   */
  private static boolean isStrictPrefixOfMarkup(final String suffix, final List<String> markers) {
    return markers.stream()
      .anyMatch(marker -> marker.startsWith(suffix) && !marker.equals(suffix));
  }

  /**
   * Nested think-block remainder: extra thinking, answer text before the next open tag, still-open flag.
   */
  private record Remainder(String thinking, String answer, boolean open) {
  }
}
