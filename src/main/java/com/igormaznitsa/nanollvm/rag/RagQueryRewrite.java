package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.AssistantParts;
import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Isolated LLM rewrite of short RAG follow-ups into standalone search keywords.
 * Does not mutate chat history or stream to the user.
 */
final class RagQueryRewrite {

  private static final SamplingParams REWRITE_SAMPLING = new SamplingParams(0.1f, 48);
  private static final Pattern SEARCH_PREFIX = Pattern.compile("(?i)^\\s*(?:search\\s*:\\s*)?");
  private static final Pattern WRAP_QUOTES = Pattern.compile("^[\"'`]+|[\"'`]+$");

  private RagQueryRewrite() {
  }

  static String userMessage(final String priorContext, final String followUp) {
    requireNonNull(followUp, "followUp");
    String follow = followUp.strip();
    if (follow.isEmpty()) {
      throw new IllegalArgumentException("followUp must not be blank");
    }
    String prior = priorContext == null ? "" : priorContext.strip();
    if (prior.isEmpty()) {
      return """
        Rewrite the question as a short keyword search for a document index.
        Reply with only the search keywords, or NONE if nothing can be searched.

        Question: %s
        """.formatted(follow).strip();
    }
    return """
      Rewrite the follow-up as a short keyword search for a document index.
      Use Prior to resolve pronouns and missing names.
      Reply with only the search keywords, or NONE if nothing can be searched.

      Prior: %s
      Follow-up: %s
      """.formatted(prior, follow).strip();
  }

  /**
   * Parses model output into search text. Empty when the model says {@code NONE} or produces
   * nothing usable.
   */
  static Optional<String> parse(final String rawModelText) {
    if (rawModelText == null || rawModelText.isBlank()) {
      return Optional.empty();
    }
    String answer = AssistantParts.parse(rawModelText).answer().strip();
    if (answer.isEmpty()) {
      return Optional.empty();
    }
    String firstLine = answer.lines()
      .map(String::strip)
      .filter(line -> !line.isEmpty())
      .findFirst()
      .orElse("");
    if (firstLine.isEmpty()) {
      return Optional.empty();
    }
    String cleaned = WRAP_QUOTES.matcher(SEARCH_PREFIX.matcher(firstLine).replaceFirst(""))
      .replaceAll("")
      .strip();
    if (cleaned.isEmpty()) {
      return Optional.empty();
    }
    if ("none".equals(cleaned.toLowerCase(Locale.ROOT))) {
      return Optional.empty();
    }
    if (cleaned.length() > 240) {
      cleaned = cleaned.substring(0, 240).strip();
    }
    return Optional.of(cleaned);
  }

  /**
   * Runs one isolated generate (thinking off) and parses the reply into search keywords.
   */
  static Optional<String> rewrite(final LLM llm, final String priorContext, final String followUp) {
    requireNonNull(llm, "llm");
    String user = userMessage(priorContext, followUp);
    List<ChatMessage> turn = List.of(ChatMessage.user(user));
    String prompt = llm.tokenizer().applyChatTemplate(
      ChatMessages.toTemplateMaps(turn), true, false);
    String raw = llm.generate(List.of(prompt), REWRITE_SAMPLING).getFirst().text();
    return parse(raw);
  }
}
