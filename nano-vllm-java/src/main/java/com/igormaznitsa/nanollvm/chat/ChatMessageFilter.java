package com.igormaznitsa.nanollvm.chat;

/**
 * Decides, for one message, whether the session keeps it and whether the model sees it.
 *
 * <p>Register one or more with {@link ChatSession#filter(ChatMessageFilter...)}. Every filter
 * must allow a destination or the message is left out of that destination. Filters run when a
 * message is added and again at the start of the next {@link ChatSession#send}, so a later change
 * can drop a message already stored.
 *
 * <pre>{@code
 * session.filter(
 *     message -> message.content().contains("secret")
 *         ? ChatAdmission.neither()
 *         : ChatAdmission.both(),
 *     message -> message.role() == ChatRole.SYSTEM
 *         ? ChatAdmission.historyOnly()
 *         : ChatAdmission.both());
 * }</pre>
 *
 * @since 1.5.0
 */
@FunctionalInterface
public interface ChatMessageFilter {

  /**
   * Admission for {@code message}. Return {@link ChatAdmission#both()} to keep the usual turn.
   * The session rejects a {@code null} return.
   *
   * @param message the turn being added or replayed; never {@code null}
   * @return history and model flags for this message; never {@code null}
   */
  ChatAdmission admit(ChatMessage message);
}
