package com.igormaznitsa.nanollvm.prompts;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Advisor wording: prompt for advisors, and mix of their notes into the facts block.
 */
public final class AdvisorPrompts {

  public static final String ROLE_PRACTICAL =
    "Practical: a few short sentences with concrete facts.";

  public static final String ROLE_ABSTRACT =
    "Abstract: a few short sentences on themes.";

  public static final String ROLE_CONSEQUENCE =
    "Consequence: a few short sentences on outcomes or next steps.";

  public static final String FOR_ADVISOR =
    "Reply in 1–2 short sentences. Prefer the facts above when present.";

  public static final String EMPTY_NOTE_FALLBACK = "I have no idea on this question.";

  public static final List<String> ADVISOR_NAMES = List.of(
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega");

  public static final Pattern ABSTAIN_REPLY = Pattern.compile(
    "(?i)^\\s*" + Pattern.quote(RagPrompts.ABSTAIN_REPLY) + "\\.?\\s*$");

  private AdvisorPrompts() {
  }

  public static String forAdvisor(final String rolePrompt) {
    requireNonNull(rolePrompt, "rolePrompt");
    String role = rolePrompt.strip();
    if (role.isEmpty()) {
      throw new IllegalArgumentException("rolePrompt must not be blank");
    }
    return role + "\n\n" + FOR_ADVISOR;
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
    if (question.isEmpty()) {
      question = base;
      facts = "";
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
    return mergedFacts + "\n\n" + question;
  }

  public static String mixNoteLine(final String note) {
    requireNonNull(note, "note");
    return "- " + note.strip().replace('"', '\'');
  }

  public static String advisorName(final int index) {
    if (index < 0) {
      throw new IllegalArgumentException("index must be >= 0");
    }
    if (index < ADVISOR_NAMES.size()) {
      return ADVISOR_NAMES.get(index);
    }
    return "Advisor" + (index + 1);
  }

  public static boolean isCounselorNameOnly(final String text) {
    if (text == null || text.isBlank()) {
      return false;
    }
    String body = text.strip();
    if (body.length() > 24) {
      return false;
    }
    String lower = body.toLowerCase(Locale.ROOT);
    return ADVISOR_NAMES.stream().anyMatch(name -> name.toLowerCase(Locale.ROOT).equals(lower))
      || lower.matches("advisor\\d+");
  }
}
