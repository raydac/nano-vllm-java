package com.igormaznitsa.nanollvm.prompts;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Chat helpers that stay model-agnostic. The library never injects system or advisor prose;
 * callers set {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#systemPrompt(String)} when they
 * want a policy. Demo wording and setup-boilerplate filters live in {@code nano-vllm-java-samples}.
 */
public final class ChatPrompts {

  private ChatPrompts() {
  }

  /**
   * Library default system text: always empty.
   */
  public static String systemFor(final Tokenizer tokenizer) {
    return "";
  }

  /**
   * Pass-through for a caller-supplied system string (null → empty). The library does not append
   * advisor prose; applications that want an advisor-aware system cue include it themselves.
   */
  public static String withAdvisorGuidance(final String systemPrompt,
                                           final boolean advisorsEnabled) {
    return systemPrompt == null ? "" : systemPrompt;
  }

  /**
   * Turn-based chat templates have no system role: fold non-blank system text into the first user turn.
   */
  public static String foldSystemIntoFirstUser(final String system, final String userContent,
                                               final boolean firstUser) {
    final String content = userContent == null ? "" : userContent;
    if (!firstUser || system == null || system.isBlank()) {
      return content;
    }
    return system + "\n\n" + content;
  }
}
