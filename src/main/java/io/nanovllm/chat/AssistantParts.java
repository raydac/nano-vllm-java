package io.nanovllm.chat;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record AssistantParts(String thinking, String answer, boolean thinkOpen) {

  private static final Pattern SPECIAL_TOKEN = Pattern.compile("<\\|[^\\s|>]*\\|>?");
  private static final Pattern LEADING_ASSISTANT = Pattern.compile("(?i)^\\s*assistant\\s*:?\\s*");
  private static final Pattern STATED_ANSWER = Pattern.compile(
      "(?i)(?:answer(?:\\s+should)?\\s+be|returns?|result(?:\\s+is)?)\\s*[:\"']?\\s*(.+)$");
  private static final List<String> CHAT_MARKUP = List.of(
      "<|im_end|>", "<|im_start|>", "<|endoftext|>",
      "<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>",
      "<think>", "</think>"
  );

  public static AssistantParts parse(String raw) {
    if (raw == null || raw.isBlank()) {
      return new AssistantParts("", "", false);
    }

    String text = holdIncompleteMarkupSuffix(raw);
    int startThink = text.indexOf("<think>");
    int endThink = startThink >= 0
        ? text.indexOf("</think>", startThink + "<think>".length())
        : text.indexOf("</think>");

    String thinking;
    boolean open;
    String after;
    if (startThink >= 0 && endThink > startThink) {
      thinking = text.substring(startThink + "<think>".length(), endThink);
      Remainder rest = splitRemainder(text.substring(endThink + "</think>".length()));
      thinking = joinThink(thinking, rest.thinking());
      after = rest.answer();
      open = rest.open();
    } else if (startThink >= 0) {
      thinking = text.substring(startThink + "<think>".length());
      after = "";
      open = true;
    } else if (endThink >= 0) {
      Remainder rest = splitRemainder(text.substring(endThink + "</think>".length()));
      thinking = rest.thinking();
      after = rest.answer();
      open = rest.open();
    } else {
      thinking = "";
      after = text;
      open = false;
    }

    return new AssistantParts(stripChatMarkup(thinking), sanitizeAnswer(after), open);
  }

  private static Remainder splitRemainder(String after) {
    after = holdIncompleteMarkupSuffix(after);
    int start = after.indexOf("<think>");
    if (start < 0) {
      return new Remainder("", after, false);
    }

    String before = after.substring(0, start);
    String fromThink = after.substring(start + "<think>".length());
    int end = fromThink.indexOf("</think>");
    if (end < 0) {
      return new Remainder(fromThink, before, true);
    }

    String inner = fromThink.substring(0, end);
    Remainder nested = splitRemainder(fromThink.substring(end + "</think>".length()));
    return new Remainder(
        joinThink(inner, nested.thinking()),
        joinAnswer(before, nested.answer()),
        nested.open());
  }

  private static String joinThink(String first, String second) {
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

  private static String joinAnswer(String first, String second) {
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

  private static String sanitizeAnswer(String text) {
    text = holdIncompleteMarkupSuffix(text);
    for (String marker : CHAT_MARKUP) {
      int at = text.indexOf(marker);
      if (at >= 0) {
        text = text.substring(0, at);
      }
    }
    return finishSanitize(text);
  }

  public static String stripChatMarkup(String text) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    String s = holdIncompleteMarkupSuffix(text);
    for (String marker : CHAT_MARKUP) {
      s = s.replace(marker, "");
    }
    return finishSanitize(s);
  }

  private static String finishSanitize(String text) {
    text = SPECIAL_TOKEN.matcher(text).replaceAll("");
    return LEADING_ASSISTANT.matcher(text).replaceFirst("").strip();
  }

  static String stripSpecialTokens(String raw) {
    return stripChatMarkup(raw);
  }

  public static String cleanAssistantText(String raw) {
    AssistantParts parts = parse(raw);
    String answer = parts.answer();
    return answer.isEmpty() ? salvageFromThinking(parts.thinking()) : answer;
  }

  public static String streamDisplayText(String raw) {
    return cleanAssistantText(raw);
  }

  public static String salvageFromThinking(String thinking) {
    if (thinking == null || thinking.isBlank()) {
      return "";
    }

    String[] lines = thinking.strip().split("\\R");
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

  private static boolean isBriefAnswerLine(String line) {
    return line.length() <= 32 && !looksLikeReasoning(line);
  }

  private static boolean looksLikeReasoning(String line) {
    String lower = line.toLowerCase();
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
  static String holdIncompleteMarkupSuffix(String text) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    int at = text.lastIndexOf('<');
    if (at < 0) {
      return text;
    }
    String suffix = text.substring(at);
    return isStrictPrefixOfMarkup(suffix) ? text.substring(0, at) : text;
  }

  private static boolean isStrictPrefixOfMarkup(String suffix) {
    for (String marker : CHAT_MARKUP) {
      if (marker.startsWith(suffix) && !marker.equals(suffix)) {
        return true;
      }
    }
    return false;
  }

  private record Remainder(String thinking, String answer, boolean open) {
  }
}
