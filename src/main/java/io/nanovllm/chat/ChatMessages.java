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
    history.add(message("system", ChatPrompts.systemFor(gemmaChat)));
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
    while (history.size() > 2) {
      String prompt = tokenizer.applyChatTemplate(history, true, enableThinking);
      if (tokenizer.encode(prompt).size() <= budget) {
        return;
      }
      history.remove(1);
      if (history.size() > 1 && "assistant".equals(history.get(1).get("role"))) {
        history.remove(1);
      }
    }
  }
}
