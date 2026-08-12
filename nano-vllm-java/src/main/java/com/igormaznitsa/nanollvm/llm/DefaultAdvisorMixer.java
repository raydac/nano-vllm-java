package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.chat.ChatHistory;
import java.util.List;

/**
 * Built-in mixer: inserts non-empty advisor notes into the facts block of {@code prompt}.
 * History is unused (grounding / salvage stay in {@link AdvisorRunner}).
 */
enum DefaultAdvisorMixer implements LlmAdvisorMixer {
  INSTANCE;

  @Override
  public String mixPrompt(
    final LLM source,
    final List<AdvisorResponse> advisorResponses,
    final ChatHistory history,
    final String prompt
  ) {
    List<String> notes = advisorResponses.stream()
      .map(AdvisorResponse::text)
      .toList();
    return AdvisorPrompt.mix(prompt, notes);
  }
}
