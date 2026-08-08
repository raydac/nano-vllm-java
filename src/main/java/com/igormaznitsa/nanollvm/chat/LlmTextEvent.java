package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

/**
 * One chat-stream text event.
 *
 * <p>For {@link LlmTextKind#TEXT_THINKING} / {@link LlmTextKind#TEXT_ASSISTANT}, {@link #text()} is a
 * delta chunk unless {@link #snapshot()} is {@code true} (full current buffer after a revise).
 * {@link #slot()} is the 1-based advisor index for {@link LlmTextKind#TEXT_ADVISOR_NOTE}; otherwise
 * {@code -1}.
 */
public record LlmTextEvent(LlmTextKind kind, String text, int slot, boolean snapshot) {

  public LlmTextEvent {
    requireNonNull(kind, "kind");
    text = text == null ? "" : text;
  }

  public static LlmTextEvent of(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, -1, false);
  }

  public static LlmTextEvent snapshot(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, -1, true);
  }

  public static LlmTextEvent advisorNote(final int slot, final String text) {
    if (slot < 1) {
      throw new IllegalArgumentException("slot must be >= 1, got " + slot);
    }
    return new LlmTextEvent(LlmTextKind.TEXT_ADVISOR_NOTE, text, slot, false);
  }
}
