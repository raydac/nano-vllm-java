package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Result of the optional advisor pass: mixed user text plus named advisor replies.
 *
 * <p>{@link com.igormaznitsa.nanollvm.chat.ChatSession} and {@link LLM#runAdvisors} use
 * {@link #modelUserText()} as the user string in the chat template for the main generate.
 * {@link #responses()} is one entry per configured advisor (empty when none ran).
 * {@link #salvageNotes()} are grounded notes kept for {@link #answerSalvageNotes()} when the
 * main answer is unusable.
 *
 * <p>Lists are unmodifiable copies. Immutable; safe to share across threads.
 *
 * @param modelUserText user text after {@link LlmAdvisorMixer#mixPrompt}; never {@code null}
 * @param responses     one {@link AdvisorResponse} per configured advisor, in configuration order
 * @param salvageNotes  notes selected for fallback (may be empty; then
 *                      {@link #answerSalvageNotes()} falls back to response texts)
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

  /**
   * No advisors ran (or none were configured): {@code modelUserText} is returned unchanged, with
   * empty {@link #responses()} and {@link #salvageNotes()}. Chat / RAG use this as a cheap skip.
   *
   * @param modelUserText prepared user string; must not be {@code null}
   * @return enrichment that leaves the turn prompt as-is
   * @throws NullPointerException if {@code modelUserText} is {@code null}
   */
  public static AdvisorEnrichment passthrough(final String modelUserText) {
    return new AdvisorEnrichment(
      requireNonNull(modelUserText, "modelUserText"), List.of(), List.of());
  }

  /**
   * {@code true} when at least one {@link #responses()} entry has a non-blank {@link AdvisorResponse#text()}.
   * Empty or whitespace-only notes do not count — mix / salvage may still be empty.
   *
   * @return whether any advisor wrote usable text
   */
  public boolean hasAdvisorNotes() {
    return this.responses.stream()
      .map(AdvisorResponse::text)
      .anyMatch(note -> note != null && !note.isBlank());
  }

  /**
   * Notes usable when the main answer is unusable: prefer {@link #salvageNotes()}, else raw
   * advisor replies. Blank entries are dropped; order is preserved; duplicates removed.
   *
   * @return unmodifiable list, possibly empty
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
