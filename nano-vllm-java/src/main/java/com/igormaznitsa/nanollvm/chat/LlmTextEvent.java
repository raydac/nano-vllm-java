package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

/**
 * One chat-stream text event.
 *
 * <p>For {@link LlmTextKind#TEXT_THINKING} / {@link LlmTextKind#TEXT_ASSISTANT}, {@link #text()} is a
 * delta chunk unless {@link #snapshot()} is {@code true} (full current buffer after a revise).
 * {@link LlmTextKind#TEXT_DEBUG} is always a full snapshot (e.g. prepared model user after advisor
 * mix). {@link #advisorName()} is set for {@link LlmTextKind#TEXT_ADVISOR_NOTE}; otherwise empty.
 */
public record LlmTextEvent(LlmTextKind kind, String text, String advisorName, boolean snapshot) {

  public LlmTextEvent {
    requireNonNull(kind, "kind");
    text = text == null ? "" : text;
    advisorName = advisorName == null ? "" : advisorName;
  }

  public static LlmTextEvent of(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, "", false);
  }

  public static LlmTextEvent snapshot(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, "", true);
  }

  public static LlmTextEvent advisorNote(final String advisorName, final String text) {
    String name = requireNonNull(advisorName, "advisorName").strip();
    if (name.isEmpty()) {
      throw new IllegalArgumentException("advisorName must not be blank");
    }
    return new LlmTextEvent(LlmTextKind.TEXT_ADVISOR_NOTE, text, name, false);
  }

  public static LlmTextEvent debug(final String text) {
    return new LlmTextEvent(LlmTextKind.TEXT_DEBUG, text, "", true);
  }
}
