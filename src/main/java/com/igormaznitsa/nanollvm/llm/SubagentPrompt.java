package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Builds isolated subagent chat prompts and mixes advisor answers into the main user text.
 */
public final class SubagentPrompt {

  private SubagentPrompt() {
  }

  /**
   * Isolated turn: subagent role as system (Gemma folds it into the first user turn) plus the
   * same prepared user text the main model will see — no conversation history.
   */
  public static String isolated(
    final Tokenizer tokenizer,
    final String rolePrompt,
    final String modelUserText
  ) {
    requireNonNull(tokenizer, "tokenizer");
    requireNonNull(rolePrompt, "rolePrompt");
    requireNonNull(modelUserText, "modelUserText");
    String role = rolePrompt.strip();
    String user = modelUserText.strip();
    if (role.isEmpty()) {
      throw new IllegalArgumentException("rolePrompt must not be blank");
    }
    if (user.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    List<ChatMessage> turn = new ArrayList<>(ChatMessages.newConversation(role));
    turn.add(ChatMessage.user(user));
    return tokenizer.applyChatTemplate(ChatMessages.toTemplateMaps(turn), true, false);
  }

  /**
   * Appends non-blank advisor notes after {@code modelUserText}. Empty answers leave the text
   * unchanged.
   */
  public static String mix(final String modelUserText, final List<String> answers) {
    return mix(modelUserText, answers, false);
  }

  /**
   * @param compact shorter trailer so tiny models (Gemma) keep answering the question instead of
   *                latching onto meta-instructions
   */
  public static String mix(
    final String modelUserText,
    final List<String> answers,
    final boolean compact
  ) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(answers, "answers");
    String base = modelUserText.strip();
    if (base.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    List<String> notes = usableNotes(answers);
    if (notes.isEmpty()) {
      return base;
    }

    String numbered = IntStream.range(0, notes.size())
      .mapToObj(i -> "[%d] %s".formatted(i + 1, notes.get(i)))
      .collect(joining("\n"));

    if (compact) {
      return base + """


        Notes:
        """ + numbered + """

        Final answer (one short sentence):
        """;
    }

    return base + """


      Advisor notes (for your answer; do not quote verbatim unless useful):
      """ + numbered;
  }

  static List<String> usableNotes(final List<String> answers) {
    return answers.stream()
      .map(answer -> answer == null ? "" : answer.strip())
      .filter(note -> !note.isEmpty())
      .toList();
  }
}
