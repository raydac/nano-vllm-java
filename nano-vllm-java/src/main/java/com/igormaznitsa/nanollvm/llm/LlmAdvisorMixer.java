package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.chat.ChatHistory;
import java.util.List;

/**
 * Merges advisor replies into the main user prompt before the primary generate.
 *
 * <p>Default mixer ({@link #defaults()}) inserts useful notes into the facts block; pass a custom
 * mixer as the first argument of {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)}.
 */
@FunctionalInterface
public interface LlmAdvisorMixer {

  static LlmAdvisorMixer defaults() {
    return DefaultAdvisorMixer.INSTANCE;
  }

  /**
   * Builds the main-user prompt from advisor replies and the current turn.
   *
   * @param source           engine that produced the advisor replies
   * @param advisorResponses one entry per configured advisor (name + generated text)
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
