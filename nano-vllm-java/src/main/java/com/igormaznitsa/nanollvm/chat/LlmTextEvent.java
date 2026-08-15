package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

/**
 * One text or status payload delivered to {@link LlmListener#onText}.
 *
 * <p>Switch on {@link #kind()} first. For {@link LlmTextKind#TEXT_THINKING} /
 * {@link LlmTextKind#TEXT_ASSISTANT} / {@link LlmTextKind#TEXT_RAW}, {@link #text()} is a
 * <em>delta</em> unless {@link #snapshot()} is {@code true} (full current buffer after a revise).
 * {@link LlmTextKind#TEXT_DEBUG} is always a full snapshot (prepared model-user after advisor mix).
 * {@link #advisorName()} is set only for {@link LlmTextKind#TEXT_ADVISOR_NOTE}; otherwise empty.
 * Null {@code text} / {@code advisorName} become {@code ""}. Immutable; safe to share.
 *
 * @param kind        channel this payload belongs to; never {@code null}
 * @param text        payload body (delta or snapshot per {@code kind} / {@code snapshot}); never
 *                    {@code null}
 * @param advisorName configured {@link com.igormaznitsa.nanollvm.llm.LlmAdvisor#name()} for
 *                    advisor notes; empty for every other kind
 * @param snapshot    {@code true} when {@code text} is the full current buffer rather than a
 *                    suffix to append
 */
public record LlmTextEvent(LlmTextKind kind, String text, String advisorName, boolean snapshot) {

  public LlmTextEvent {
    requireNonNull(kind, "kind");
    text = text == null ? "" : text;
    advisorName = advisorName == null ? "" : advisorName;
  }

  /**
   * Delta event with no advisor name ({@code snapshot == false}).
   *
   * @param kind channel; must not be {@code null}
   * @param text payload; {@code null} becomes {@code ""}
   */
  public static LlmTextEvent of(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, "", false);
  }

  /**
   * Full-buffer event ({@code snapshot == true}), for a revise that replaces earlier deltas.
   *
   * @param kind channel; must not be {@code null}
   * @param text full current buffer; {@code null} becomes {@code ""}
   */
  public static LlmTextEvent snapshot(final LlmTextKind kind, final String text) {
    return new LlmTextEvent(kind, text, "", true);
  }

  /**
   * Advisor-note event ({@link LlmTextKind#TEXT_ADVISOR_NOTE}, delta).
   *
   * @param advisorName non-blank advisor name
   * @param text        note body; {@code null} becomes {@code ""}
   * @throws IllegalArgumentException if {@code advisorName} is blank after strip
   */
  public static LlmTextEvent advisorNote(final String advisorName, final String text) {
    String name = requireNonNull(advisorName, "advisorName").strip();
    if (name.isEmpty()) {
      throw new IllegalArgumentException("advisorName must not be blank");
    }
    return new LlmTextEvent(LlmTextKind.TEXT_ADVISOR_NOTE, text, name, false);
  }

  /**
   * Debug snapshot ({@link LlmTextKind#TEXT_DEBUG}) — typically the prepared model-user string.
   *
   * @param text debug payload; {@code null} becomes {@code ""}
   */
  public static LlmTextEvent debug(final String text) {
    return new LlmTextEvent(LlmTextKind.TEXT_DEBUG, text, "", true);
  }
}
