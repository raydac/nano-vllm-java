package com.igormaznitsa.nanollvm.chat;

public record ChatReply(String thinking, String answer, boolean thinkOpen) {

  public ChatReply {
    thinking = thinking == null ? "" : thinking;
    answer = answer == null ? "" : answer;
  }

  public static ChatReply from(AssistantParts parts) {
    return new ChatReply(parts.thinking(), parts.answer(), parts.thinkOpen());
  }

  public String text() {
    return this.answer;
  }
}
