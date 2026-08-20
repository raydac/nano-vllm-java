package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

/**
 * One advisor generate result: the configured {@link LlmAdvisor#name()} plus the generated note.
 *
 * <p>Produced by the optional advisor pass <em>before</em> the main chat generate. Mixers read
 * {@link #text()}; session salvage may reuse it when the main answer is unusable.
 * {@code text} is the answer-channel parse of the advisor completion (thinking stripped), never
 * {@code null} — empty when the advisor wrote nothing or the note filter dropped it. Immutable;
 * safe to share.
 *
 * @param advisorName unique non-blank name from {@link LlmAdvisor#name()}
 * @param text        generated advisor note after answer-channel parse; never {@code null}
 */
public record AdvisorResponse(String advisorName, String text) {

  public AdvisorResponse {
    requireNonNull(advisorName, "advisorName");
    String name = advisorName.strip();
    if (name.isEmpty()) {
      throw new IllegalArgumentException("advisorName must not be blank");
    }
    advisorName = name;
    text = text == null ? "" : text;
  }
}
