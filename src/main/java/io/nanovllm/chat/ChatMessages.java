package io.nanovllm.chat;

import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ChatMessages {

  private static final int PROMPT_MARGIN = 16;

  private ChatMessages() {
  }

  public static List<ChatMessage> newConversation(boolean gemmaChat) {
    return newConversation(ChatPrompts.systemFor(gemmaChat));
  }

  public static List<ChatMessage> newConversation(String systemPrompt) {
    List<ChatMessage> history = new ArrayList<>();
    if (systemPrompt != null && !systemPrompt.isBlank()) {
      history.add(ChatMessage.system(systemPrompt));
    }
    return history;
  }

  public static List<Map<String, String>> toTemplateMaps(List<ChatMessage> history) {
    return history.stream().map(ChatMessage::toMap).toList();
  }

  public static void truncateHistory(
      List<ChatMessage> history,
      Tokenizer tokenizer,
      int maxModelLen,
      int maxTokens
  ) {
    int budget = Math.max(64, maxModelLen - maxTokens - PROMPT_MARGIN);
    boolean enableThinking = !tokenizer.isGemmaChat();
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

  public static void scrubSetupBoilerplateTurns(List<ChatMessage> history) {
    for (int i = 0; i < history.size(); i++) {
      ChatMessage msg = history.get(i);
      if (msg.role() == ChatRole.ASSISTANT && ChatPrompts.isSetupBoilerplate(msg.content())) {
        history.set(i, ChatMessage.assistant("Hello!"));
      }
    }
  }
}
