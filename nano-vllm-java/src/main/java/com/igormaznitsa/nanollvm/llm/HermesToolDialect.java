package com.igormaznitsa.nanollvm.llm;

import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;
import java.util.Map;

/**
 * Qwen3 / Qwen2.5 Hermes tool calls: a JSON object inside {@code <tool_call>}.
 *
 * <p>The vocabulary tokens {@code <tool_call>}, {@code </tool_call>}, {@code <tool_response>}, and
 * {@code </tool_response>} are ordinary pieces ({@code special=false} on Qwen3). The turn still
 * ends on {@code <|im_end|>}. Results go back as a user turn, which is what the Qwen3 chat
 * template does with a {@code tool} role.
 */
final class HermesToolDialect implements ToolDialect {

  private static final String PREAMBLE = """
    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    %s
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>""";

  private static ToolCall parseBody(final String body) {
    int start = body.indexOf('{');
    int end = body.lastIndexOf('}');
    if (start < 0 || end <= start) {
      return new ToolCall("", "{}");
    }
    try {
      Map<String, Object> object = Json.parseObject(body.substring(start, end + 1));
      Object name = object.get("name");
      if (!(name instanceof String function) || function.isBlank()) {
        return new ToolCall("", "{}");
      }
      return new ToolCall(function, ToolJson.argumentsObject(object.get("arguments")));
    } catch (IllegalArgumentException ex) {
      return new ToolCall("", "{}");
    }
  }

  @Override
  public String id() {
    return "hermes";
  }

  @Override
  public List<String> extraStopTokens() {
    return List.of();
  }

  @Override
  public String render(
    final List<ChatMessage> history,
    final List<ModelTool> tools,
    final Tokenizer tokenizer,
    final boolean enableThinking,
    final ThinkTags tags
  ) {
    String specs = tools.stream().map(ToolJson::openAiFunction).collect(joining("\n"));
    return ChatMlTools.render(history, PREAMBLE.formatted(specs), false, tokenizer, enableThinking,
      tags);
  }

  @Override
  public List<ToolCall> parseCalls(final String assistantText) {
    return ToolMarkup.bodies(assistantText == null ? "" : assistantText, ChatMlTools.OPEN,
        ChatMlTools.CLOSE)
      .stream()
      .map(HermesToolDialect::parseBody)
      .toList();
  }

  @Override
  public String visibleText(final String assistantText) {
    return ChatMlTools.visible(assistantText);
  }

  @Override
  public List<ChatMessage> appendExchange(
    final String assistantText,
    final List<ToolCall> calls,
    final List<String> results
  ) {
    return ChatMlTools.appendExchange(assistantText, calls, results);
  }
}
