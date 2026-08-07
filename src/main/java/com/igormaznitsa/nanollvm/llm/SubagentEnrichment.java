package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Result of the optional subagent pass: enriched user text, raw advisor notes for the thinking
 * stream, and the filtered notes actually mixed into the main prompt.
 */
public record SubagentEnrichment(
  String modelUserText,
  List<String> advisorNotes,
  List<String> groundedNotes
) {

  public SubagentEnrichment {
    requireNonNull(modelUserText, "modelUserText");
    advisorNotes = List.copyOf(requireNonNull(advisorNotes, "advisorNotes"));
    groundedNotes = List.copyOf(requireNonNull(groundedNotes, "groundedNotes"));
  }

  public static SubagentEnrichment passthrough(final String modelUserText) {
    return new SubagentEnrichment(
      requireNonNull(modelUserText, "modelUserText"), List.of(), List.of());
  }

  public boolean hasAdvisorNotes() {
    return this.advisorNotes.stream().anyMatch(note -> note != null && !note.isBlank());
  }

  public boolean hasGroundedNotes() {
    return this.groundedNotes.stream().anyMatch(note -> note != null && !note.isBlank());
  }

  public int groundedMixedCount() {
    return (int) this.groundedNotes.stream()
      .map(note -> note == null ? "" : note.strip())
      .filter(note -> !note.isEmpty())
      .count();
  }
}
