package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.Map;

/**
 * One chat turn: a {@link ChatRole} and text content.
 *
 * <p>Use the factories ({@link #system}, {@link #user}, {@link #assistant}) when building history
 * for {@link ChatSession} or a chat template. {@link ChatSession#send} stores the user turn and
 * later the visible {@link ChatReply#answer()} as an assistant message — thinking is not kept
 * here. Null {@code content} becomes {@code ""}. Immutable; safe to share across threads.
 *
 * <pre>{@code
 * List<ChatMessage> seed = List.of(
 *     ChatMessage.system("Be brief."),
 *     ChatMessage.user("What is 2+2?"));
 * }</pre>
 *
 * @param role    speaker for this turn; never {@code null}
 * @param content turn text; never {@code null} (empty when the caller passed {@code null})
 */
public record ChatMessage(ChatRole role, String content) {

  /**
   * Canonical constructor: {@code role} must be non-null; null {@code content} becomes {@code ""}.
   */
  public ChatMessage {
    requireNonNull(role, "role");
    content = content == null ? "" : content;
  }

  /**
   * System / instruction turn (optional seed at the start of a conversation).
   *
   * @param content instruction text; {@code null} becomes {@code ""}
   * @return system message
   */
  public static ChatMessage system(final String content) {
    return new ChatMessage(ChatRole.SYSTEM, content);
  }

  /**
   * User turn.
   *
   * @param content user text; {@code null} becomes {@code ""}
   * @return user message
   */
  public static ChatMessage user(final String content) {
    return new ChatMessage(ChatRole.USER, content);
  }

  /**
   * Assistant turn (visible answer only in session history).
   *
   * @param content assistant text; {@code null} becomes {@code ""}
   * @return assistant message
   */
  public static ChatMessage assistant(final String content) {
    return new ChatMessage(ChatRole.ASSISTANT, content);
  }

  /**
   * Rebuilds a message from a chat-template map ({@code role} / {@code content} keys).
   *
   * <p>Unknown or missing {@code role} becomes {@link ChatRole#USER}. Missing {@code content}
   * becomes {@code ""}.
   *
   * @param map wire map; must not be {@code null}
   * @return parsed message
   * @throws NullPointerException if {@code map} is {@code null}
   */
  public static ChatMessage fromMap(final Map<String, String> map) {
    requireNonNull(map, "map");
    return new ChatMessage(
      ChatRole.fromWire(map.get("role")),
      map.getOrDefault("content", ""));
  }

  /**
   * Chat-template map: {@code role} is {@link ChatRole#wireName()}, {@code content} is the text.
   *
   * @return an unmodifiable two-entry map
   */
  public Map<String, String> toMap() {
    return Map.of(
      "role", this.role.wireName(),
      "content", this.content);
  }
}
