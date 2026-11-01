package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Qwen3.5 tool calls, including the 0.8B checkpoint. The official template (and SGLang's
 * {@code qwen3_coder} parser) asks for an XML function inside {@code <tool_call>}, not the
 * Hermes JSON object Qwen3 uses. A JSON object in the same span is still accepted.
 */
final class Qwen35ToolDialect implements ToolDialect {

  private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

  private static final String PREAMBLE = """
    # Tools

    You have access to the following functions:

    <tools>
    %s
    </tools>

    If you choose to call a function ONLY reply in the following format with NO suffix:

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    <parameter=example_parameter_2>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>
    </tool_call>

    Reminder:
    - Function calls MUST follow the specified format: an inner <function=...> block must be nested within <tool_call></tool_call> XML tags
    - Required parameters MUST be specified
    - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls""";

  private static List<ToolCall> parseFunctions(final String body) {
    List<ToolCall> calls = new ArrayList<>();
    int cursor = 0;
    while (cursor < body.length()) {
      int start = body.indexOf("<function=", cursor);
      if (start < 0) {
        break;
      }
      int nameEnd = body.indexOf('>', start);
      int close = body.indexOf("</function>", start);
      if (nameEnd < 0 || close < 0) {
        calls.add(new ToolCall("", "{}"));
        break;
      }
      String name = body.substring(start + "<function=".length(), nameEnd).strip();
      String inner = body.substring(nameEnd + 1, close);
      calls.add(NAME.matcher(name).matches()
        ? new ToolCall(name, parametersObject(inner))
        : new ToolCall("", "{}"));
      cursor = close + "</function>".length();
    }
    if (calls.isEmpty()) {
      calls.add(new ToolCall("", "{}"));
    }
    return calls;
  }

  private static String parametersObject(final String inner) {
    Map<String, Object> arguments = new LinkedHashMap<>();
    int cursor = 0;
    while (cursor < inner.length()) {
      int start = inner.indexOf("<parameter=", cursor);
      if (start < 0) {
        break;
      }
      int nameEnd = inner.indexOf('>', start);
      int close = inner.indexOf("</parameter>", start);
      if (nameEnd < 0 || close < 0 || nameEnd > close) {
        break;
      }
      String name = inner.substring(start + "<parameter=".length(), nameEnd).strip();
      String raw = inner.substring(nameEnd + 1, close).strip();
      if (NAME.matcher(name).matches()) {
        arguments.put(name, parameterValue(raw));
      }
      cursor = close + "</parameter>".length();
    }
    return ToolJson.write(arguments);
  }

  private static Object parameterValue(final String raw) {
    if (raw.equals("true") || raw.equals("false")) {
      return Boolean.valueOf(raw);
    }
    if (raw.equals("null")) {
      return null;
    }
    if (looksNumber(raw)) {
      try {
        return Json.parse(raw);
      } catch (IllegalArgumentException ignored) {
        return raw;
      }
    }
    if ((raw.startsWith("{") && raw.endsWith("}")) || (raw.startsWith("[") && raw.endsWith("]"))) {
      try {
        return Json.parse(raw);
      } catch (IllegalArgumentException ignored) {
        return raw;
      }
    }
    return raw;
  }

  private static boolean looksNumber(final String raw) {
    int index = raw.startsWith("-") ? 1 : 0;
    if (index >= raw.length() || !Character.isDigit(raw.charAt(index))) {
      return false;
    }
    for (int i = index; i < raw.length(); i++) {
      char ch = raw.charAt(i);
      if (!Character.isDigit(ch) && ch != '.' && ch != 'e' && ch != 'E' && ch != '+' && ch != '-') {
        return false;
      }
    }
    return true;
  }

  private static ToolCall parseJson(final String body) {
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
    return "qwen3.5";
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
    String specs =
      tools.stream().map(ToolJson::openAiFunction).reduce((left, right) -> left + "\n" + right)
        .orElse("");
    return ChatMlTools.render(history, PREAMBLE.formatted(specs), true, tokenizer, enableThinking,
      tags);
  }

  @Override
  public List<ToolCall> parseCalls(final String assistantText) {
    String text = assistantText == null ? "" : assistantText;
    List<ToolCall> calls = new ArrayList<>();
    for (String body : ToolMarkup.bodies(text, ChatMlTools.OPEN, ChatMlTools.CLOSE)) {
      if (body.contains("<function=")) {
        calls.addAll(parseFunctions(body));
      } else {
        calls.add(parseJson(body));
      }
    }
    return calls;
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
