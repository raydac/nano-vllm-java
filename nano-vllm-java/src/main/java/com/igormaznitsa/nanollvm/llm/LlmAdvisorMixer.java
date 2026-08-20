package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.chat.ChatHistory;
import java.util.List;

/**
 * Merges advisor replies into the main user prompt before the primary generate.
 *
 * <p>After every configured {@link LlmAdvisor} has produced a note, the engine calls
 * {@link #mixPrompt} once. The returned string is what the chat template sees as the latest user
 * turn (often already RAG-augmented). Implementations may insert notes into a facts block, rewrite
 * the turn, or ignore some replies.
 *
 * <p>{@link #defaults()} inserts non-empty notes into the facts section of {@code prompt} and
 * ignores history (grounding / salvage stay in the advisor runner). Pass a custom mixer as the
 * first argument of {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)} when the default
 * placement is wrong for the application. The mixed text must not be blank.
 *
 * @see LlmAdvisor
 * @see AdvisorEnrichment
 */
@FunctionalInterface
public interface LlmAdvisorMixer {

  /**
   * Built-in mixer: copies non-empty {@link AdvisorResponse#text()} into the facts block of
   * {@code prompt}. Dialog history is unused. Empty-only replies leave the prompt unchanged.
   *
   * @return a shared mixer instance; safe to reuse
   */
  static LlmAdvisorMixer defaults() {
    return DefaultAdvisorMixer.INSTANCE;
  }

  /**
   * Builds the user string for the main chat generate from advisor replies and the current turn.
   *
   * <p>{@code prompt} is the prepared model-user text (question, optional RAG context). History is
   * prior dialog the mixer may consult; the default mixer ignores it. Return a non-blank user
   * string — blank mixes fail the turn.
   *
   * @param source           engine that produced the advisor replies
   * @param advisorResponses one entry per configured advisor, in configuration order
   * @param history          prior dialog available for mix decisions (may be empty)
   * @param prompt           prepared model-user text for this turn (often RAG-augmented)
   * @return user text for the main chat generate; must not be blank
   */
  String mixPrompt(
    LLM source,
    List<AdvisorResponse> advisorResponses,
    ChatHistory history,
    String prompt);
}
