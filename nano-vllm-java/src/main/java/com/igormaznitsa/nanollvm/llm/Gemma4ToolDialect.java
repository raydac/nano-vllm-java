package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;
import java.util.Map;

/**
 * Gemma 4 tool turns. Six tokens bound the lifecycle: {@code <|tool>} / {@code <tool|>} declare a
 * function, {@code <|tool_call>} / {@code <tool_call|>} request a call, and
 * {@code <|tool_response>} / {@code <tool_response|>} carry the result. String values use
 * {@code <|"|>}. The result stays inside the same model turn, and {@code <|tool_response>} is an
 * extra stop so the model does not invent the result.
 */
final class Gemma4ToolDialect implements ToolDialect {

  private static final String CALL_OPEN = "<|tool_call>";
  private static final String CALL_CLOSE = "<tool_call|>";
  private static final String RESPONSE_OPEN = "<|tool_response>";
  private static final String RESPONSE_CLOSE = "<tool_response|>";

  private static String declaration(final ModelTool tool) {
    return "<|tool>declaration:" + tool.name()
      + "{description:" + GemmaValues.quote(tool.description())
      + ",parameters:" + GemmaValues.format(Json.parse(tool.parametersSchema()), true)
      + "}<tool|>";
  }

  private static ToolCall parseBody(final String body) {
    String text = body.strip();
    if (!text.startsWith("call:")) {
      return new ToolCall("", "{}");
    }
    int brace = text.indexOf('{');
    if (brace < 0) {
      return new ToolCall("", "{}");
    }
    String name = text.substring("call:".length(), brace).strip();
    try {
      Object value = GemmaValues.parse(text.substring(brace));
      if (!(value instanceof Map<?, ?>)) {
        return new ToolCall("", "{}");
      }
      return new ToolCall(name, ToolJson.write(value));
    } catch (IllegalArgumentException ex) {
      return new ToolCall("", "{}");
    }
  }

  private static String responseFields(final String result) {
    String text = result == null ? "" : result.strip();
    if (text.startsWith("{") && text.endsWith("}")) {
      try {
        String object = GemmaValues.format(Json.parseObject(text), false);
        return object.substring(1, object.length() - 1);
      } catch (IllegalArgumentException ignored) {
        return "result:" + GemmaValues.quote(result == null ? "" : result);
      }
    }
    return "result:" + GemmaValues.quote(result == null ? "" : result);
  }

  private static String systemText(final List<ChatMessage> history) {
    if (history.isEmpty() || history.getFirst().role() != ChatRole.SYSTEM) {
      return "";
    }
    return history.getFirst().content().strip();
  }

  @Override
  public String id() {
    return "gemma4";
  }

  @Override
  public List<String> extraStopTokens() {
    return List.of(RESPONSE_OPEN);
  }

  @Override
  public String render(
    final List<ChatMessage> history,
    final List<ModelTool> tools,
    final Tokenizer tokenizer,
    final boolean enableThinking,
    final ThinkTags tags
  ) {
    requireNonNull(tags, "tags");
    StringBuilder prompt = new StringBuilder();
    if (tokenizer.tokenId("<bos>").isPresent()) {
      prompt.append("<bos>");
    }
    String system = systemText(history);
    String declarations =
      tools.stream().map(Gemma4ToolDialect::declaration).reduce("", String::concat);
    prompt.append("<|turn>system\n");
    if (enableThinking) {
      prompt.append("<|think|>\n");
    }
    if (!system.isEmpty()) {
      prompt.append(system);
    }
    prompt.append(declarations).append("<turn|>\n");

    boolean continued = false;
    for (ChatMessage message : history) {
      if (message.role() == ChatRole.SYSTEM) {
        continue;
      }
      String role = message.role() == ChatRole.ASSISTANT ? "model" : message.role().wireName();
      boolean openTurn = message == history.getLast()
        && message.role() == ChatRole.ASSISTANT
        && message.content().contains(RESPONSE_CLOSE);
      prompt.append("<|turn>").append(role).append('\n').append(message.content().strip());
      if (openTurn) {
        continued = true;
      } else {
        prompt.append("<turn|>\n");
      }
    }
    if (!continued) {
      prompt.append("<|turn>model\n");
    }
    return prompt.toString();
  }

  @Override
  public List<ToolCall> parseCalls(final String assistantText) {
    String text = assistantText == null ? "" : assistantText;
    return ToolMarkup.bodies(text, CALL_OPEN, CALL_CLOSE).stream()
      .map(Gemma4ToolDialect::parseBody)
      .toList();
  }

  @Override
  public String visibleText(final String assistantText) {
    return ToolMarkup.withoutTurnEnds(ToolMarkup.hide(assistantText, CALL_OPEN, CALL_CLOSE))
      .replace(RESPONSE_CLOSE, "")
      .strip();
  }

  @Override
  public List<ChatMessage> appendExchange(
    final String assistantText,
    final List<ToolCall> calls,
    final List<String> results
  ) {
    ToolExchange.requirePaired(calls, results);
    StringBuilder turn = new StringBuilder(ToolMarkup.throughLastClose(
      ToolMarkup.withoutTurnEnds(assistantText), CALL_OPEN, CALL_CLOSE));
    for (int i = 0; i < calls.size() && i < results.size(); i++) {
      String name = calls.get(i).name().isBlank() ? "tool" : calls.get(i).name();
      turn.append(RESPONSE_OPEN)
        .append("response:")
        .append(name)
        .append('{')
        .append(responseFields(results.get(i)))
        .append('}')
        .append(RESPONSE_CLOSE);
    }
    return List.of(ChatMessage.assistant(turn.toString()));
  }
}
