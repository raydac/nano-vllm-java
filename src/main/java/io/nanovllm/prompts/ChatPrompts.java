package io.nanovllm.prompts;

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

  private static final Pattern SETUP_BOILERPLATE = Pattern.compile(
      "(?i).*\\b(i(?:'m| am) ready|let'?s begin|okay[,.]?\\s*i understand)\\b.*"
  );

  private ChatPrompts() {
  }

  public static String systemFor(boolean gemmaChat) {
    return gemmaChat ? GEMMA_CHAT_SYSTEM : CHAT_SYSTEM;
  }

  public static String gemmaUserContent(String system, String userContent, boolean firstUser) {
    if (userContent == null) {
      userContent = "";
    }
    if (!firstUser || system == null || system.isBlank()) {
      return userContent;
    }
    return system + "\n\n" + userContent;
  }

  public static boolean isSetupBoilerplate(String reply) {
    if (reply == null || reply.isBlank()) {
      return false;
    }
    String trimmed = reply.strip();
    if (trimmed.length() > 120) {
      return false;
    }
    return SETUP_BOILERPLATE.matcher(trimmed).matches();
  }
}
