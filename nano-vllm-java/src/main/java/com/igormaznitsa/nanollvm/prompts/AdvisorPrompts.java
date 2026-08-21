package com.igormaznitsa.nanollvm.prompts;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Structural helpers for advisor note mixing and abstain detection. Caller-owned names and role
 * prompts are supplied via {@link com.igormaznitsa.nanollvm.llm.LlmAdvisor}; this class does not
 * ship demo role text or name catalogs.
 */
public final class AdvisorPrompts {

  public static final Pattern ABSTAIN_REPLY = Pattern.compile(
    "(?i)^\\s*" + Pattern.quote(RagPrompts.ABSTAIN_REPLY) + "\\.?\\s*$");

  private AdvisorPrompts() {
  }

  /**
   * Facts first (optional), blank line, then the question.
   */
  public static String advisorUser(final String question, final String context) {
    requireNonNull(question, "question");
    String q = question.strip();
    if (q.isEmpty()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    if (context == null || context.isBlank()) {
      return q;
    }
    return RagPrompts.withContext(q, context);
  }

  /**
   * Inserts advisor note bullets into the facts block (before the blank line + user question).
   */
  public static String withGeneratedNotes(final String modelUserText, final List<String> notes) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(notes, "notes");
    String base = modelUserText.strip();
    if (base.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }
    if (notes.isEmpty()) {
      return base;
    }
    String facts = RagPrompts.facts(base);
    String question = RagPrompts.question(base);
    boolean hadPassages = !facts.isBlank();
    if (question.isEmpty()) {
      question = base;
      facts = "";
      hadPassages = false;
    }
    String existingFacts = facts;
    String block = notes.stream()
      .map(AdvisorPrompts::mixNoteLine)
      .filter(line -> !existingFacts.contains(line))
      .collect(Collectors.joining("\n"));
    if (block.isEmpty()) {
      return base;
    }
    String mergedFacts = existingFacts.isBlank() ? block : existingFacts + "\n" + block;
    return hadPassages
      ? RagPrompts.withContext(question, mergedFacts)
      : mergedFacts + "\n\n" + question;
  }

  public static String mixNoteLine(final String note) {
    requireNonNull(note, "note");
    return "- " + note.strip().replace('"', '\'');
  }
}
