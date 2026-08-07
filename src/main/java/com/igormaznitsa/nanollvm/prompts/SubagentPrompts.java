package com.igormaznitsa.nanollvm.prompts;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.regex.Pattern;

/**
 * Model-facing subagent role text, grounded-extraction rules, and mix templates.
 */
public final class SubagentPrompts {

  public static final String ROLE_PRACTICAL = """
    Practical extractor: pull concrete facts, places, actions, and constraints from Context.
    For broad questions, report every concrete detail Context states, even if it is incomplete.
    Quote or paraphrase only what the text supports.
    """.strip();

  public static final String ROLE_ABSTRACT = """
    Abstract extractor: name themes, roles, and structure that are explicit in Context.
    For broad questions, name roles or themes Context actually mentions.
    Do not add symbols or places that are not written there.
    """.strip();

  public static final String ROLE_CONSEQUENCE = """
    Consequence extractor: from Context facts only, note outcomes or what happens next when Context states them.
    Do not invent new destinations, characters, or settings absent from Context.
    """.strip();

  public static final String NO_HIT_ADVISOR = """
      Indexed documents returned no passages for this Question.
      From your role, state in 1–2 short sentences what the Question asks and why the index gives nothing useful.
      Do not answer from general or world knowledge; describe the gap only.
    """.strip();

  public static final String GROUNDED_EXTRACTION = """
    You are a pre-answer advisor, not the user-facing assistant.
    Use only facts from the Context section (ignore the Question wording for new facts).
    If the Question is open-ended, report every concrete detail Context supports; a partial summary is correct.
    If Context is missing or empty, describe what you looked for and what is absent — still from your role angle.
    Do not reply with only "%s" or copy abstain instructions meant for the final answer.
    Do not invent places, names, dates, or events. At most 2 short sentences. No greetings.
    """.formatted(RagPrompts.ABSTAIN_REPLY).strip();

  public static final String MIX_CLAIMS_HEADER = """
    Unverified advisor hints (NOT ranked options; do NOT copy as the answer).
    Prefer Context over every hint. If hints disagree, ignore them all.
    """.strip();

  public static final String MIX_FULL_FOOTER = """
    Answer from Context first. Use a hint only when it restates Context wording.
    """.strip();

  public static final Pattern ABSTAIN_REPLY = Pattern.compile(
    "(?i)^\\s*" + Pattern.quote(RagPrompts.ABSTAIN_REPLY) + "\\.?\\s*$");

  /**
   * Markers that end a Context block when scanning a prepared user message.
   */
  public static final List<String> CONTEXT_BLOCK_END_MARKERS = List.of(
    "\nAnswer ",
    "\nFinal answer",
    "\nAdvisor notes",
    "\nUnverified");

  private SubagentPrompts() {
  }

  public static String[] demoRoles() {
    return demoRolesGemma();
  }

  public static String[] demoRolesGemma() {
    return new String[] {ROLE_PRACTICAL, ROLE_ABSTRACT, ROLE_CONSEQUENCE};
  }

  public static String[] demoRolesQwen() {
    return new String[] {ROLE_PRACTICAL, ROLE_ABSTRACT};
  }

  public static String groundedRole(final String rolePrompt) {
    return groundedRole(rolePrompt, false);
  }

  public static String groundedRole(final String rolePrompt, final boolean ragNoHits) {
    requireNonNull(rolePrompt, "rolePrompt");
    String role = rolePrompt.strip();
    if (role.isEmpty()) {
      throw new IllegalArgumentException("rolePrompt must not be blank");
    }
    if (ragNoHits) {
      return role + "\n\n" + NO_HIT_ADVISOR + "\n\n" + GROUNDED_EXTRACTION;
    }
    return role + "\n\n" + GROUNDED_EXTRACTION;
  }

  public static String claimLine(final String label, final String note) {
    return "- claim-%s: %s".formatted(label, note);
  }

  public static String mixCompact(final String claimsBlock, final String modelUserText) {
    return MIX_CLAIMS_HEADER + "\n" + claimsBlock + "\n\n" + MIX_FULL_FOOTER + "\n\n" +
      modelUserText;
  }

  public static String mixFull(final String modelUserText, final String claimsBlock) {
    return modelUserText + "\n\n" + MIX_CLAIMS_HEADER + "\n" + claimsBlock + "\n\n" +
      MIX_FULL_FOOTER;
  }
}
