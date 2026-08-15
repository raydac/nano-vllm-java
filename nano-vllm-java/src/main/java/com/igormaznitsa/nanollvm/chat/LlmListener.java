package com.igormaznitsa.nanollvm.chat;

import com.igormaznitsa.nanollvm.llm.LLM;

/**
 * Receives chat text and engine status events from {@link LLM}, model/RAG load, and sessions.
 *
 * <p>{@code source} may be {@code null} for load-time status (model / RAG / KV allocation) before
 * an {@link LLM} exists. Callbacks run on the calling thread (load or generate) and must return
 * quickly without blocking or re-entering {@link LLM#generate}.
 */
@FunctionalInterface
public interface LlmListener {

  /**
   * Receives one chat or status event. {@code source} is {@code null} for load-time status before an
   * {@link LLM} exists.
   *
   * @param source engine that emitted the event, or {@code null} during load
   * @param event  text or status payload; must not be {@code null}
   */
  void onText(LLM source, LlmTextEvent event);
}
