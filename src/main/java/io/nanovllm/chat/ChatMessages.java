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

  public static List<Map<String, String>> newConversation(List<String> knowledge) {
    List<Map<String, String>> history = new ArrayList<>();
    history.add(message("system", ChatPrompts.chatSystemWithKnowledge(knowledge)));
    return history;
  }

  public static void syncSystemMessage(List<Map<String, String>> history, List<String> knowledge) {
    String system = ChatPrompts.chatSystemWithKnowledge(knowledge);
    if (history.isEmpty()) {
      history.add(message("system", system));
      return;
    }
    history.getFirst().put("content", system);
  }

  public static void truncateHistory(
      List<Map<String, String>> history,
      Tokenizer tokenizer,
      int maxModelLen,
      int maxTokens
  ) {
    int budget = Math.max(64, maxModelLen - maxTokens - PROMPT_MARGIN);
    while (history.size() > 2) {
      String prompt = tokenizer.applyChatTemplate(history, true, true);
      if (tokenizer.encode(prompt).size() <= budget) {
        return;
      }
      history.remove(1);
      if (history.size() > 1 && "assistant".equals(history.get(1).get("role"))) {
        history.remove(1);
      }
    }
  }

  public static String complete(
      io.nanovllm.LLM llm,
      Tokenizer tokenizer,
      String systemPrompt,
      String userPayload,
      io.nanovllm.SamplingParams params
  ) {
    return complete(llm, tokenizer, systemPrompt, userPayload, params, null);
  }

  /**
   * @param assistantPrefix optional text appended after the chat template so the model
   *                        continues a forced format (e.g. think seed + {@code "+ "}).
   */
  public static String complete(
      io.nanovllm.LLM llm,
      Tokenizer tokenizer,
      String systemPrompt,
      String userPayload,
      io.nanovllm.SamplingParams params,
      String assistantPrefix
  ) {
    List<Map<String, String>> messages = new ArrayList<>();
    messages.add(message("system", systemPrompt));
    messages.add(message("user", userPayload));
    String prompt = tokenizer.applyChatTemplate(messages, true, false);
    if (assistantPrefix != null && !assistantPrefix.isEmpty()) {
      prompt = prompt + assistantPrefix;
    }
    List<io.nanovllm.LLM.GenerationOutput> out = llm.generate(List.of(prompt), params, false, null);
    String generated = tokenizer.decode(out.getFirst().tokenIds(), false);
    String full = (assistantPrefix == null || assistantPrefix.isEmpty())
        ? generated
        : assistantPrefix + generated;
    // Strip think blocks; if the model left facts only inside thinking, recover them.
    AssistantParts parts = AssistantParts.parse(full);
    String answer = parts.answer();
    if (isBlankOrPlusStub(answer) && parts.thinking() != null && parts.thinking().contains("+")) {
      return parts.thinking().strip();
    }
    return answer;
  }

  private static boolean isBlankOrPlusStub(String text) {
    if (text == null || text.isBlank()) {
      return true;
    }
    String t = text.strip();
    return t.equals("+") || t.equals("+ ") || t.equalsIgnoreCase("NONE");
  }

  public static void appendAssistantLine(List<Map<String, String>> history, String text) {
    System.out.print("assistant> ");
    System.out.println(text);
    System.out.println();
    history.add(message("assistant", text));
  }

  public static String oneLineSummary(String answer, int maxLen) {
    if (answer == null || answer.isBlank()) {
      return "(empty)";
    }
    String oneLine = answer.replace('\n', ' ').strip();
    return oneLine.length() <= maxLen ? oneLine : oneLine.substring(0, maxLen - 3) + "…";
  }
}
