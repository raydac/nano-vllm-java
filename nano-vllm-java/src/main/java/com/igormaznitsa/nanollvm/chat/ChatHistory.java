package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.stream.Stream;

/**
 * Immutable snapshot of chat turns for advisor mix and similar read-only consumers.
 *
 * <p>Not a live session: {@link ChatSession#history()} is a mutable list;
 * wrap a copy with {@link #of(List)} when a mixer or test needs a frozen view.
 * Empty snapshots share {@link #empty()}. Safe to share across threads after construction.
 *
 * <pre>{@code
 * ChatHistory history = ChatHistory.of(session.history());
 * ChatHistory questionsOnly = history.userTurns();
 * }</pre>
 */
public final class ChatHistory {

  private static final ChatHistory EMPTY = new ChatHistory(List.of());

  private final List<ChatMessage> messages;

  private ChatHistory(final List<ChatMessage> messages) {
    this.messages = messages;
  }

  /**
   * Shared empty snapshot.
   *
   * @return the interned empty history
   */
  public static ChatHistory empty() {
    return EMPTY;
  }

  /**
   * Frozen copy of {@code messages}. Empty input returns {@link #empty()}.
   *
   * @param messages turns in conversation order; must not be {@code null}
   * @return frozen copy, or {@link #empty()} when {@code messages} is empty
   * @throws NullPointerException if {@code messages} is {@code null}
   */
  public static ChatHistory of(final List<ChatMessage> messages) {
    requireNonNull(messages, "messages");
    if (messages.isEmpty()) {
      return EMPTY;
    }
    return new ChatHistory(List.copyOf(messages));
  }

  /**
   * Unmodifiable turns in conversation order.
   *
   * @return conversation turns; never {@code null}
   */
  public List<ChatMessage> messages() {
    return this.messages;
  }

  /**
   * {@code true} when {@link #messages()} is empty.
   *
   * @return whether this snapshot has no turns
   */
  public boolean isEmpty() {
    return this.messages.isEmpty();
  }

  /**
   * Number of turns in {@link #messages()}.
   *
   * @return turn count
   */
  public int size() {
    return this.messages.size();
  }

  /**
   * Stream over {@link #messages()} (not parallel).
   *
   * @return sequential stream of turns
   */
  public Stream<ChatMessage> stream() {
    return this.messages.stream();
  }

  /**
   * Copy containing only {@link ChatRole#USER} turns, in original order.
   *
   * @return frozen user-only snapshot
   */
  public ChatHistory userTurns() {
    return of(this.messages.stream()
      .filter(message -> message.role() == ChatRole.USER)
      .toList());
  }
}
