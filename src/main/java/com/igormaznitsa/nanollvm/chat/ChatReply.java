package com.igormaznitsa.nanollvm.chat;

import com.igormaznitsa.nanollvm.llm.GenerationStats;

/**
 * Parsed assistant turn: optional thinking scratchpad, visible answer, and engine stats for the
 * main generate that produced this turn ({@link GenerationStats#NONE} while streaming).
 *
 * <p>Immutable value type; safe to share across threads after construction.
 */
public record ChatReply(
  String thinking,
  String answer,
  boolean thinkOpen,
  GenerationStats stats
) {

  public ChatReply {
    thinking = thinking == null ? "" : thinking;
    answer = answer == null ? "" : answer;
    stats = stats == null ? GenerationStats.NONE : stats;
  }

  /**
   * Streaming / parse helper without measured stats.
   */
  public ChatReply(final String thinking, final String answer, final boolean thinkOpen) {
    this(thinking, answer, thinkOpen, GenerationStats.NONE);
  }

  static ChatReply from(final AssistantParts parts) {
    return new ChatReply(parts.thinking(), parts.answer(), parts.thinkOpen());
  }

  public static ChatReply parse(final String raw) {
    return from(AssistantParts.parse(raw));
  }

  public static String cleanAssistantText(final String raw) {
    return AssistantParts.cleanAssistantText(raw);
  }

  public static String streamDisplayText(final String raw) {
    return AssistantParts.streamDisplayText(raw);
  }

  public static String salvageFromThinking(final String thinking) {
    return AssistantParts.salvageFromThinking(thinking);
  }

  public static String stripChatMarkup(final String text) {
    return AssistantParts.stripChatMarkup(text);
  }

  public ChatReply withStats(final GenerationStats generateStats) {
    return new ChatReply(this.thinking, this.answer, this.thinkOpen, generateStats);
  }

  public String text() {
    return this.answer;
  }
}
