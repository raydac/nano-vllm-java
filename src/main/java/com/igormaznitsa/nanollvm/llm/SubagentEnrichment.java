package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Result of the optional subagent pass: enriched user text plus the advisor notes that were mixed
 * in (empty when no subagents ran or all answers were discarded).
 */
public record SubagentEnrichment(String modelUserText, List<String> advisorNotes) {

  public SubagentEnrichment {
    requireNonNull(modelUserText, "modelUserText");
    advisorNotes = List.copyOf(requireNonNull(advisorNotes, "advisorNotes"));
  }

  public static SubagentEnrichment passthrough(final String modelUserText) {
    return new SubagentEnrichment(requireNonNull(modelUserText, "modelUserText"), List.of());
  }

  public boolean hasAdvisorNotes() {
    return !this.advisorNotes.isEmpty();
  }
}
