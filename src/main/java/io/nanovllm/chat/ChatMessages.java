package io.nanovllm.chat;

import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ChatMessages {

  private static final int PROMPT_MARGIN = 16;

  private ChatMessages() {
  }

  public static Map<String, String> message(String role, String content) {
    Map<String, String> m = new LinkedHashMap<>();
    m.put("role", role);
    m.put("content", content);
    return m;
  }

  public static List<Map<String, String>> newConversation(boolean gemmaChat) {
    List<Map<String, String>> history = new ArrayList<>();
    String system = ChatPrompts.systemFor(gemmaChat);
    if (system != null && !system.isBlank()) {
      history.add(message("system", system));
    }
    return history;
  }

  public static void truncateHistory(
      List<Map<String, String>> history,
      Tokenizer tokenizer,
      int maxModelLen,
      int maxTokens
  ) {
    int budget = Math.max(64, maxModelLen - maxTokens - PROMPT_MARGIN);
    boolean enableThinking = !tokenizer.isGemmaChat();
    int minKeep = !history.isEmpty() && "system".equals(history.getFirst().get("role")) ? 2 : 1;
    while (history.size() > minKeep) {
      String prompt = tokenizer.applyChatTemplate(history, true, enableThinking);
      if (tokenizer.encode(prompt).size() <= budget) {
        return;
      }
      int dropAt = "system".equals(history.getFirst().get("role")) ? 1 : 0;
      if (dropAt >= history.size()) {
        return;
      }
      history.remove(dropAt);
      if (dropAt < history.size() && "assistant".equals(history.get(dropAt).get("role"))) {
        history.remove(dropAt);
      }
    }
  }

  public static void scrubSetupBoilerplateTurns(List<Map<String, String>> history) {
    for (Map<String, String> msg : history) {
      if ("assistant".equals(msg.get("role"))
          && ChatPrompts.isSetupBoilerplate(msg.getOrDefault("content", ""))) {
        msg.put("content", "Hello!");
      }
    }
  }
}
