package com.igormaznitsa.nanollvm.samples.utils;

import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.regex.Pattern;

/**
 * Demo-only chat system policies and turn-based recovery helpers. Not part of the published
 * library — tune per sample / model family here.
 *
 * @since 1.1.0
 */
public final class SampleChatPrompts {

  /**
   * Thinking-vocab demos: short dialog rules + think format.
   */
  public static final String QWEN_THINKING_SYSTEM = """
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
   * Short ChatML cue for plain assistant demos (no {@code <think>} format rules).
   */
  public static final String PLAIN_ASSISTANT_SYSTEM = """
    You are a helpful assistant. Reply briefly and clearly to the user's message.
    Do not invent facts. If you do not know, say so.
    """.strip();
  /**
   * Turn-based demos (e.g. Gemma): top-k 64 matches common generation_config defaults.
   */
  public static final int TURN_BASED_TOP_K = 64;
  private static final String ARCH_LFM2 = "lfm2";
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

  private SampleChatPrompts() {
  }

  /**
   * Demo system policy: turn-based → empty; think-vocab → {@link #QWEN_THINKING_SYSTEM}; LFM2 →
   * {@link #PLAIN_ASSISTANT_SYSTEM}; other ChatML → empty so the model's own template can apply.
   */
  public static String forDemo(final String architectureName, final Tokenizer tokenizer) {
    if (tokenizer == null || tokenizer.isTurnBasedChat()) {
      return "";
    }
    if (tokenizer.invitesThinking()) {
      return QWEN_THINKING_SYSTEM;
    }
    if (ARCH_LFM2.equals(architectureName)) {
      return PLAIN_ASSISTANT_SYSTEM;
    }
    return "";
  }

  /**
   * Sampling knobs for demos: turn-based chat uses top-k {@link #TURN_BASED_TOP_K}; others leave
   * top-k off.
   */
  public static SamplingParams samplingForDemo(final Tokenizer tokenizer, final int maxTokens) {
    int topK = tokenizer != null && tokenizer.isTurnBasedChat() ? TURN_BASED_TOP_K : 0;
    return SamplingParams.builder()
      .temperature(0.6f)
      .maxTokens(maxTokens)
      .topK(topK)
      .build();
  }

  /**
   * Short setup / role acknowledgments that small advisor demos often emit instead of an answer.
   */
  public static boolean isSetupBoilerplate(final String reply) {
    if (reply == null || reply.isBlank()) {
      return true;
    }
    String trimmed = reply.strip();
    if (trimmed.length() > SETUP_BOILERPLATE_MAX_LEN) {
      return false;
    }
    return SETUP_BOILERPLATE.matcher(trimmed).matches();
  }
}
