package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.stream.Stream;

/**
 * Immutable snapshot of chat turns for advisor mix and similar read-only consumers.
 */
public final class ChatHistory {

  private static final ChatHistory EMPTY = new ChatHistory(List.of());

  private final List<ChatMessage> messages;

  private ChatHistory(final List<ChatMessage> messages) {
    this.messages = messages;
  }

  public static ChatHistory empty() {
    return EMPTY;
  }

  public static ChatHistory of(final List<ChatMessage> messages) {
    requireNonNull(messages, "messages");
    if (messages.isEmpty()) {
      return EMPTY;
    }
    return new ChatHistory(List.copyOf(messages));
  }

  public List<ChatMessage> messages() {
    return this.messages;
  }

  public boolean isEmpty() {
    return this.messages.isEmpty();
  }

  public int size() {
    return this.messages.size();
  }

  public Stream<ChatMessage> stream() {
    return this.messages.stream();
  }

  public ChatHistory userTurns() {
    return of(this.messages.stream()
      .filter(message -> message.role() == ChatRole.USER)
      .toList());
  }
}
