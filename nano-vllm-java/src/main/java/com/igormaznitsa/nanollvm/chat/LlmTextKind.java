package com.igormaznitsa.nanollvm.chat;

/**
 * Kind of text or status carried by an {@link LlmTextEvent}.
 *
 * <p>{@code TEXT_*} kinds are chat/stream content; {@code STATUS_*} kinds are progress / info
 * lines from load and optional generate progress.
 */
public enum LlmTextKind {
  /**
   * Chain-of-thought / thinking scratchpad fragment.
   */
  TEXT_THINKING,
  /**
   * Visible assistant answer fragment.
   */
  TEXT_ASSISTANT,
  /**
   * Unparsed tokenizer decode of completion tokens so far (think tags and chat specials kept).
   * Deltas follow the same suffix/snapshot rules as {@link #TEXT_THINKING} / {@link #TEXT_ASSISTANT}.
   *
   * @since 1.1.0
   */
  TEXT_RAW,
  /**
   * Isolated advisor note (thinking stream).
   */
  TEXT_ADVISOR_NOTE,
  /**
   * Session diagnostics (salvage, empty-reply fallback, …).
   */
  TEXT_DIAGNOSTICS,
  /**
   * Debug payload (e.g. prepared model user text after advisor mix). Emitted only when
   * {@link ChatSession#emitDebugPrompts(boolean)} is {@code true}.
   */
  TEXT_DEBUG,
  /**
   * Informational status line (load, warmup, …).
   */
  STATUS_INFO,
  /**
   * Progress status line (batch generate tqdm-style).
   */
  STATUS_PROGRESS
}
