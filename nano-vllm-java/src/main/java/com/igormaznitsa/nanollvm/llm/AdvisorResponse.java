package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

/**
 * One advisor generate result: configured {@link LlmAdvisor#name()} plus the generated text.
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
