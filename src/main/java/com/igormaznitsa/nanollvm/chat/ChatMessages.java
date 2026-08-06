package com.igormaznitsa.nanollvm.chat;

import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public final class ChatMessages {

  private static final int PROMPT_MARGIN = 16;

  private ChatMessages() {
  }

  public static List<ChatMessage> newConversation(final boolean gemmaChat) {
    return newConversation(ChatPrompts.systemFor(gemmaChat));
  }

  public static List<ChatMessage> newConversation(final String systemPrompt) {
    List<ChatMessage> history = new ArrayList<>();
    if (systemPrompt != null && !systemPrompt.isBlank()) {
      history.add(ChatMessage.system(systemPrompt));
    }
    return history;
  }

  public static List<Map<String, String>> toTemplateMaps(final List<ChatMessage> history) {
    return history.stream().map(ChatMessage::toMap).toList();
  }

  public static void truncateHistory(
      final List<ChatMessage> history,
      final Tokenizer tokenizer,
      final int maxModelLen,
      final int maxTokens
  ) {
    int budget = Math.max(64, maxModelLen - maxTokens - PROMPT_MARGIN);
    boolean enableThinking = tokenizer.invitesThinking();
    int minKeep = !history.isEmpty() && history.getFirst().role() == ChatRole.SYSTEM ? 2 : 1;
    while (history.size() > minKeep) {
      String prompt = tokenizer.applyChatTemplate(toTemplateMaps(history), true, enableThinking);
      if (tokenizer.encode(prompt).size() <= budget) {
        return;
      }
      int dropAt = history.getFirst().role() == ChatRole.SYSTEM ? 1 : 0;
      if (dropAt >= history.size()) {
        return;
      }
      history.remove(dropAt);
      if (dropAt < history.size() && history.get(dropAt).role() == ChatRole.ASSISTANT) {
        history.remove(dropAt);
      }
    }
  }

  public static void scrubSetupBoilerplateTurns(final List<ChatMessage> history) {
    IntStream.range(0, history.size())
        .filter(i -> history.get(i).role() == ChatRole.ASSISTANT
            && ChatPrompts.isSetupBoilerplate(history.get(i).content()))
        .forEach(i -> history.set(i, ChatMessage.assistant("Hello!")));
  }
}
