package com.igormaznitsa.nanollvm.prompts;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.regex.Pattern;

public final class ChatPrompts {

  public static final String CHAT_SYSTEM = """
      You are the Assistant. The human is the User. Never swap those roles.

      Answer from the conversation history in this session.
      Style: one short, new reply that answers THIS turn.
      - Vary wording each turn; do not reuse the same opener or stock line.
      - Never reply with fillers like "Okay, I'm ready", "Let's begin", or "How can I help you?"
        when the User asked something concrete — answer the ask instead.
      - Do not greet again if you already greeted.
      - Do not repeat the User's words back as your reply.
      - Do not invent facts. If unknown from the conversation, say you don't know.

      Thinking format (use for non-trivial replies):
      - Start with <think> … </think>, then the user-visible answer.
      - Inside think, keep 2–4 short lines: user intent, useful context from history, reply plan.
      - Always close </think> before the answer. Never leave thinking open.
      - Never put the final user-facing sentence only inside <think>.
      """.strip();

  /**
   * Gemma IT has no system role. Long instructions glued onto the first user turn
   * push 270M into setup-boilerplate ("Okay, I'm ready") that then latches multi-turn.
   * Keep empty — put any guidance in the user text only when needed.
   */
  public static final String GEMMA_CHAT_SYSTEM = "";

  /**
   * LFM2 (and other non-Qwen ChatML models): short role cue without {@code <think>} format
   * rules — those instructions make the model narrate the template instead of answering.
   */
  public static final String PLAIN_CHAT_SYSTEM = """
      You are a helpful assistant. Reply briefly and clearly to the user's message.
      Do not invent facts. If you do not know, say so.
      """.strip();

  /**
   * Extra system guidance when advisors are configured. Kept short; not applied to Gemma
   * (blank system — guidance lives in the mixed user turn instead).
   */
  public static final String ADVISOR_AWARE_ADDON = """
    Advisor hints may appear in the user turn. Answer the user's topic.
    Use hints as optional perspective. Never narrate advisor roles or copy instruction text.
    """.strip();

  private static final Pattern SETUP_BOILERPLATE = Pattern.compile(
    "(?i).*\\b("
      + "i(?:'m| am) ready"
      + "|let'?s begin"
      + "|okay[,.]?\\s*i understand"
      + "|i will not invent"
      + "|i will respond"
      + "|short viewpoint"
      + "|user-facing assistant"
      + "|pre-answer advisor"
      + "|from my role"
      + "|explanation of my role"
      + "|these instructions"
      + ")\\b.*"
  );

  private static final int SETUP_BOILERPLATE_MAX_LEN = 240;

  private ChatPrompts() {
  }

  public static String systemFor(final Tokenizer tokenizer) {
    return systemFor(tokenizer, false);
  }

  public static String systemFor(final Tokenizer tokenizer, final boolean advisorsEnabled) {
    String base;
    if (tokenizer == null || tokenizer.isGemmaChat()) {
      base = GEMMA_CHAT_SYSTEM;
    } else if (!tokenizer.invitesThinking()) {
      base = PLAIN_CHAT_SYSTEM;
    } else {
      base = CHAT_SYSTEM;
    }
    return withAdvisorGuidance(base, advisorsEnabled);
  }

  public static String systemFor(final boolean gemmaChat) {
    return gemmaChat ? GEMMA_CHAT_SYSTEM : CHAT_SYSTEM;
  }

  public static String withAdvisorGuidance(final String systemPrompt,
                                           final boolean advisorsEnabled) {
    if (!advisorsEnabled) {
      return systemPrompt == null ? "" : systemPrompt;
    }
    String base = systemPrompt == null ? "" : systemPrompt.strip();
    if (base.isEmpty()) {
      return "";
    }
    return base + "\n\n" + ADVISOR_AWARE_ADDON;
  }

  public static String gemmaUserContent(final String system, final String userContent,
                                        final boolean firstUser) {
    final String content = userContent == null ? "" : userContent;
    if (!firstUser || system == null || system.isBlank()) {
      return content;
    }
    return system + "\n\n" + content;
  }

  public static boolean isSetupBoilerplate(final String reply) {
    if (reply == null || reply.isBlank()) {
      return false;
    }
    String trimmed = reply.strip();
    if (trimmed.length() > SETUP_BOILERPLATE_MAX_LEN) {
      return false;
    }
    return SETUP_BOILERPLATE.matcher(trimmed).matches();
  }
}
