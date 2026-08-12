package com.igormaznitsa.nanollvm.samples.utils;

/**
 * Demo advisor role strings and system-prompt add-ons. Not part of the published library —
 * callers build {@link com.igormaznitsa.nanollvm.llm.LlmAdvisor} with their own name + prompt.
 */
public final class SampleAdvisorPrompts {

  public static final String FOR_ADVISOR =
    "Reply in 1–2 short sentences. Prefer the facts above when present.";

  public static final String ROLE_PRACTICAL =
    "Practical: a few short sentences with concrete facts.\n\n" + FOR_ADVISOR;

  public static final String ROLE_ABSTRACT =
    "Abstract: a few short sentences on themes.\n\n" + FOR_ADVISOR;

  public static final String ROLE_CONSEQUENCE =
    "Consequence: a few short sentences on outcomes or next steps.\n\n" + FOR_ADVISOR;

  /**
   * Extra system guidance when the Example wires advisors and already has a non-blank system
   * prompt (skipped when the demo system string is empty, e.g. turn-based chats).
   */
  public static final String ADVISOR_AWARE_ADDON = """
    Advisor hints may appear in the user turn. Answer the user's topic.
    Use hints as optional perspective. Never narrate advisor roles or copy instruction text.
    """.strip();

  private SampleAdvisorPrompts() {
  }

  public static String withAdvisorAddon(final String systemPrompt) {
    String base = systemPrompt == null ? "" : systemPrompt.strip();
    if (base.isEmpty()) {
      return "";
    }
    return base + "\n\n" + ADVISOR_AWARE_ADDON;
  }
}
