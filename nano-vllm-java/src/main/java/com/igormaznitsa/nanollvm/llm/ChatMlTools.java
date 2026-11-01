package com.igormaznitsa.nanollvm.llm;

import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * ChatML tool turns shared by Qwen3 (Hermes JSON) and Qwen3.5 (XML function calls).
 * Both families wrap the result in {@code <tool_response>} inside a user turn.
 */
final class ChatMlTools {

  static final String OPEN = "<tool_call>";
  static final String CLOSE = "</tool_call>";

  private ChatMlTools() {
  }

  static String render(
    final List<ChatMessage> history,
    final String toolBlock,
    final boolean toolsFirst,
    final Tokenizer tokenizer,
    final boolean enableThinking,
    final ThinkTags tags
  ) {
    return tokenizer.applyChatTemplate(
      ChatMessages.toTemplateMaps(withToolSystem(history, toolBlock, toolsFirst)),
      true,
      enableThinking,
      tags.open(),
      tags.close());
  }

  static String visible(final String assistantText) {
    return ToolMarkup.withoutTurnEnds(ToolMarkup.hide(assistantText, OPEN, CLOSE)).strip();
  }

  static List<ChatMessage> appendExchange(
    final String assistantText,
    final List<ToolCall> calls,
    final List<String> results
  ) {
    ToolExchange.requirePaired(calls, results);
    String body = results.stream()
      .map(result -> "<tool_response>\n" + result + "\n</tool_response>")
      .collect(joining("\n"));
    return List.of(
      ChatMessage.assistant(ToolMarkup.throughLastClose(
        ToolMarkup.withoutTurnEnds(assistantText), OPEN, CLOSE)),
      ChatMessage.user(body));
  }

  private static List<ChatMessage> withToolSystem(
    final List<ChatMessage> history,
    final String toolBlock,
    final boolean toolsFirst
  ) {
    List<ChatMessage> copy = new ArrayList<>(history.size() + 1);
    if (!history.isEmpty() && history.getFirst().role() == ChatRole.SYSTEM) {
      String system = history.getFirst().content().strip();
      String merged = system.isEmpty()
        ? toolBlock
        : toolsFirst ? toolBlock + "\n\n" + system : system + "\n\n" + toolBlock;
      copy.add(ChatMessage.system(merged));
      copy.addAll(history.subList(1, history.size()));
      return copy;
    }
    copy.add(ChatMessage.system(toolBlock));
    copy.addAll(history);
    return copy;
  }
}
