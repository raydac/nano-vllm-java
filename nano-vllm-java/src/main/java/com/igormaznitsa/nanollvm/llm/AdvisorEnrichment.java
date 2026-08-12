package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Result of the optional advisor pass: mixed user text plus named advisor replies.
 */
public record AdvisorEnrichment(
  String modelUserText,
  List<AdvisorResponse> responses,
  List<String> salvageNotes
) {

  public AdvisorEnrichment {
    requireNonNull(modelUserText, "modelUserText");
    responses = List.copyOf(requireNonNull(responses, "responses"));
    salvageNotes = List.copyOf(requireNonNull(salvageNotes, "salvageNotes"));
  }

  public static AdvisorEnrichment passthrough(final String modelUserText) {
    return new AdvisorEnrichment(
      requireNonNull(modelUserText, "modelUserText"), List.of(), List.of());
  }

  public boolean hasAdvisorNotes() {
    return this.responses.stream()
      .map(AdvisorResponse::text)
      .anyMatch(note -> note != null && !note.isBlank());
  }

  /**
   * Notes usable when the main answer is unusable: prefer grounded salvage notes, else raw
   * advisor replies (blank entries dropped).
   */
  public List<String> answerSalvageNotes() {
    List<String> source = this.salvageNotes.isEmpty()
      ? this.responses.stream().map(AdvisorResponse::text).toList()
      : this.salvageNotes;
    return source.stream()
      .map(note -> note == null ? "" : note.strip())
      .filter(note -> !note.isEmpty())
      .distinct()
      .toList();
  }
}
