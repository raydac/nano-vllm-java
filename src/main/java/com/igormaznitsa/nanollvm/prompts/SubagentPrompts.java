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
    Quote or paraphrase only what the text supports.
    """.strip();

  public static final String ROLE_ABSTRACT = """
    Abstract extractor: name themes, roles, and structure that are explicit in Context.
    Do not add symbols or places that are not written there.
    """.strip();

  public static final String ROLE_CONSEQUENCE = """
    Consequence extractor: from Context facts only, note what could follow next in-story.
    Do not invent new destinations, characters, or settings absent from Context.
    """.strip();

  public static final String GROUNDED_EXTRACTION = """
    Use only facts from any Context in the user message.
    If Context does not contain the answer, reply exactly: %s.
    Do not invent places, names, or events. At most 2 short sentences. No greetings.
    """.formatted(RagPrompts.ABSTAIN_REPLY).strip();

  public static final String MIX_CLAIMS_HEADER = """
    Unverified advisor claims (NOT ranked options; do NOT copy as the answer).
    If claims disagree with each other or with Context, ignore ALL claims.
    """.strip();

  public static final String MIX_FULL_FOOTER = """
    Answer from Context first. Use a claim only when it restates Context.
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
    return new String[] {ROLE_PRACTICAL, ROLE_ABSTRACT, ROLE_CONSEQUENCE};
  }

  public static String groundedRole(final String rolePrompt) {
    requireNonNull(rolePrompt, "rolePrompt");
    String role = rolePrompt.strip();
    if (role.isEmpty()) {
      throw new IllegalArgumentException("rolePrompt must not be blank");
    }
    return role + "\n\n" + GROUNDED_EXTRACTION;
  }

  public static String claimLine(final String label, final String note) {
    return "- claim-%s: %s".formatted(label, note);
  }

  public static String mixCompact(final String claimsBlock, final String modelUserText) {
    return MIX_CLAIMS_HEADER + "\n" + claimsBlock + "\n\n" + modelUserText;
  }

  public static String mixFull(final String modelUserText, final String claimsBlock) {
    return modelUserText + "\n\n" + MIX_CLAIMS_HEADER + "\n" + claimsBlock + "\n\n" +
      MIX_FULL_FOOTER;
  }
}
