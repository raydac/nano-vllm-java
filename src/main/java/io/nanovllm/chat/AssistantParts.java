package io.nanovllm.chat;

import java.util.List;
import java.util.regex.Pattern;

public record AssistantParts(String thinking, String answer, boolean thinkOpen) {

  private static final Pattern SPECIAL_TOKEN = Pattern.compile("<\\|[^\\s|>]*\\|>?");
  private static final Pattern LEADING_ASSISTANT = Pattern.compile("(?i)^\\s*assistant\\s*:?\\s*");
  private static final List<String> CHAT_MARKUP = List.of(
      "<|im_end|>", "<|im_start|>", "<|endoftext|>",
      "<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>",
      "<think>", "</think>"
  );

  public static AssistantParts parse(String raw) {
    if (raw == null || raw.isBlank()) {
      return new AssistantParts("", "", false);
    }
    String text = raw;
    int startThink = text.indexOf("<think>");
    int endThink = text.lastIndexOf("</think>");

    String thinking;
    boolean open;
    String after;
    if (startThink >= 0 && endThink > startThink) {
      thinking = text.substring(startThink + "<think>".length(), endThink);
      after = text.substring(endThink + "</think>".length());
      open = false;
    } else if (startThink >= 0) {
      thinking = text.substring(startThink + "<think>".length());
      after = "";
      open = true;
    } else if (endThink >= 0) {
      thinking = "";
      after = text.substring(endThink + "</think>".length());
      open = false;
    } else {
      thinking = "";
      after = text;
      open = false;
    }

    return new AssistantParts(stripChatMarkup(thinking), sanitizeAnswer(after), open);
  }

  private static String sanitizeAnswer(String text) {
    for (String marker : CHAT_MARKUP) {
      int at = text.indexOf(marker);
      if (at >= 0) {
        text = text.substring(0, at);
      }
    }
    return finishSanitize(text);
  }

  /**
   * Remove Gemma/Qwen chat template tokens from streamed decode (incremental or final).
   */
  public static String stripChatMarkup(String text) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    String s = text;
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
    return answer.isEmpty() ? parts.thinking().strip() : answer;
  }

  public static String streamDisplayText(String raw) {
    return cleanAssistantText(raw);
  }
}
