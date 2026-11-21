package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

/**
 * Whether one chat message is kept in the session transcript and whether it is sent to the model.
 *
 * <p>Several {@link ChatMessageFilter}s are combined with {@link #and}: a destination stays open
 * only when every filter allows it. {@link #both()} is the session default when no filter is set.
 *
 * <pre>{@code
 * session.filter(message -> {
 *   if (message.content().contains("secret")) {
 *     return ChatAdmission.neither();
 *   }
 *   if (message.role() == ChatRole.SYSTEM) {
 *     return ChatAdmission.historyOnly();
 *   }
 *   if (message.content().startsWith("note:")) {
 *     return ChatAdmission.modelOnly();
 *   }
 *   return ChatAdmission.both();
 * });
 * }</pre>
 *
 * @param keepInHistory {@code true} to keep the message in {@link ChatSession#history()}
 * @param sendToModel   {@code true} to include the message in the next model prompt
 * @since 1.5.0
 */
public record ChatAdmission(boolean keepInHistory, boolean sendToModel) {

  /**
   * Keep the message in the transcript and send it to the model.
   *
   * @return both destinations open
   */
  public static ChatAdmission both() {
    return new ChatAdmission(true, true);
  }

  /**
   * Keep the message in the transcript and leave it out of the model prompt.
   *
   * @return history only
   */
  public static ChatAdmission historyOnly() {
    return new ChatAdmission(true, false);
  }

  /**
   * Send the message to the model and leave it out of {@link ChatSession#history()}.
   *
   * @return model only
   */
  public static ChatAdmission modelOnly() {
    return new ChatAdmission(false, true);
  }

  /**
   * Drop the message from the transcript and from the model prompt.
   *
   * @return both destinations closed
   */
  public static ChatAdmission neither() {
    return new ChatAdmission(false, false);
  }

  /**
   * A destination stays open only when both admissions allow it.
   *
   * @param other the other filter result; must not be {@code null}
   * @return the intersection
   * @throws NullPointerException if {@code other} is {@code null}
   */
  public ChatAdmission and(final ChatAdmission other) {
    requireNonNull(other, "other");
    return new ChatAdmission(
      this.keepInHistory && other.keepInHistory,
      this.sendToModel && other.sendToModel);
  }
}
