package com.igormaznitsa.nanollvm.llm;

import static java.util.stream.Collectors.joining;

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
 * LFM2 tool calls. Definitions sit in the system turn between {@code <|tool_list_start|>} and
 * {@code <|tool_list_end|>}. The model writes a Python call list between
 * {@code <|tool_call_start|>} and {@code <|tool_call_end|>}. The result is a {@code tool} turn
 * wrapped in {@code <|tool_response_start|>} / {@code <|tool_response_end|>}.
 */
final class Lfm2ToolDialect implements ToolDialect {

  static final String CALL_OPEN = "<|tool_call_start|>";
  static final String CALL_CLOSE = "<|tool_call_end|>";
  private static final String LIST_OPEN = "<|tool_list_start|>";
  private static final String LIST_CLOSE = "<|tool_list_end|>";
  private static final String RESPONSE_OPEN = "<|tool_response_start|>";
  private static final String RESPONSE_CLOSE = "<|tool_response_end|>";
  private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

  private static List<ToolCall> parseList(final String body) {
    String text = body;
    if (text.startsWith("[")) {
      text = text.substring(1);
    }
    if (text.endsWith("]")) {
      text = text.substring(0, text.length() - 1);
    }
    List<ToolCall> calls = new ArrayList<>();
    int cursor = 0;
    while (cursor < text.length()) {
      cursor = skipSpace(text, cursor);
      if (cursor >= text.length()) {
        break;
      }
      int nameEnd = cursor;
      while (nameEnd < text.length() && isNameChar(text.charAt(nameEnd), nameEnd == cursor)) {
        nameEnd++;
      }
      String name = text.substring(cursor, nameEnd);
      int open = skipSpace(text, nameEnd);
      if (!NAME.matcher(name).matches() || open >= text.length() || text.charAt(open) != '(') {
        calls.add(new ToolCall("", "{}"));
        break;
      }
      int close = matchingParen(text, open);
      if (close < 0) {
        calls.add(new ToolCall("", "{}"));
        break;
      }
      calls.add(new ToolCall(name, argumentsObject(text.substring(open + 1, close))));
      cursor = skipSpace(text, close + 1);
      if (cursor < text.length() && text.charAt(cursor) == ',') {
        cursor++;
      }
    }
    return calls;
  }

  private static String argumentsObject(final String args) {
    Map<String, Object> arguments = new LinkedHashMap<>();
    int cursor = 0;
    while (cursor < args.length()) {
      cursor = skipSpace(args, cursor);
      if (cursor >= args.length()) {
        break;
      }
      int nameEnd = cursor;
      while (nameEnd < args.length() && isNameChar(args.charAt(nameEnd), nameEnd == cursor)) {
        nameEnd++;
      }
      String name = args.substring(cursor, nameEnd);
      int eq = skipSpace(args, nameEnd);
      if (!NAME.matcher(name).matches() || eq >= args.length() || args.charAt(eq) != '=') {
        return "{}";
      }
      int valueStart = skipSpace(args, eq + 1);
      ValueRead read = readValue(args, valueStart);
      arguments.put(name, read.value());
      cursor = skipSpace(args, read.next());
      if (cursor < args.length() && args.charAt(cursor) == ',') {
        cursor++;
      }
    }
    return ToolJson.write(arguments);
  }

  private static ValueRead readValue(final String text, final int start) {
    if (start >= text.length()) {
      return new ValueRead("", start);
    }
    char ch = text.charAt(start);
    if (ch == '\'' || ch == '"') {
      return readQuoted(text, start, ch);
    }
    if (ch == '{' || ch == '[') {
      int end = matchingBracket(text, start);
      if (end < 0) {
        return new ValueRead(text.substring(start).strip(), text.length());
      }
      String literal = text.substring(start, end + 1);
      try {
        return new ValueRead(Json.parse(literal), end + 1);
      } catch (IllegalArgumentException ex) {
        return new ValueRead(literal, end + 1);
      }
    }
    int end = start;
    while (end < text.length() && text.charAt(end) != ',' && text.charAt(end) != ')') {
      end++;
    }
    String literal = text.substring(start, end).strip();
    return new ValueRead(plainValue(literal), end);
  }

  private static Object plainValue(final String literal) {
    if (literal.equals("true") || literal.equals("false")) {
      return Boolean.valueOf(literal);
    }
    if (literal.equals("null") || literal.equals("None")) {
      return null;
    }
    try {
      return Json.parse(literal);
    } catch (IllegalArgumentException ex) {
      return literal;
    }
  }

  private static ValueRead readQuoted(final String text, final int start, final char quote) {
    StringBuilder value = new StringBuilder();
    int index = start + 1;
    while (index < text.length()) {
      char ch = text.charAt(index);
      if (ch == '\\' && index + 1 < text.length()) {
        char escaped = text.charAt(index + 1);
        value.append(switch (escaped) {
          case 'n' -> '\n';
          case 'r' -> '\r';
          case 't' -> '\t';
          default -> escaped;
        });
        index += 2;
        continue;
      }
      if (ch == quote) {
        return new ValueRead(value.toString(), index + 1);
      }
      value.append(ch);
      index++;
    }
    return new ValueRead(value.toString(), text.length());
  }

  private static int matchingParen(final String text, final int open) {
    return matching(text, open, '(', ')');
  }

  private static int matchingBracket(final String text, final int open) {
    char left = text.charAt(open);
    char right = left == '{' ? '}' : ']';
    return matching(text, open, left, right);
  }

  private static int matching(final String text, final int open, final char left,
                              final char right) {
    int depth = 0;
    boolean inSingle = false;
    boolean inDouble = false;
    for (int i = open; i < text.length(); i++) {
      char ch = text.charAt(i);
      if (ch == '\\' && (inSingle || inDouble)) {
        i++;
        continue;
      }
      if (ch == '\'' && !inDouble) {
        inSingle = !inSingle;
        continue;
      }
      if (ch == '"' && !inSingle) {
        inDouble = !inDouble;
        continue;
      }
      if (inSingle || inDouble) {
        continue;
      }
      if (ch == left) {
        depth++;
      } else if (ch == right && --depth == 0) {
        return i;
      }
    }
    return -1;
  }

  private static int skipSpace(final String text, final int index) {
    int cursor = index;
    while (cursor < text.length() && Character.isWhitespace(text.charAt(cursor))) {
      cursor++;
    }
    return cursor;
  }

  private static boolean isNameChar(final char ch, final boolean first) {
    if (first) {
      return Character.isLetter(ch) || ch == '_';
    }
    return Character.isLetterOrDigit(ch) || ch == '_';
  }

  @Override
  public String id() {
    return "lfm2";
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
    String list =
      LIST_OPEN + "[" + tools.stream().map(ToolJson::bareFunction).collect(joining(", "))
        + "]" + LIST_CLOSE;
    String block = "List of tools: " + list;
    return ChatMlTools.render(history, block, false, tokenizer, enableThinking, tags);
  }

  @Override
  public List<ToolCall> parseCalls(final String assistantText) {
    String text = assistantText == null ? "" : assistantText;
    List<ToolCall> calls = new ArrayList<>();
    for (String body : ToolMarkup.bodies(text, CALL_OPEN, CALL_CLOSE)) {
      calls.addAll(parseList(body.strip()));
    }
    return calls;
  }

  @Override
  public String visibleText(final String assistantText) {
    return ToolMarkup.withoutTurnEnds(ToolMarkup.hide(assistantText, CALL_OPEN, CALL_CLOSE))
      .strip();
  }

  @Override
  public List<ChatMessage> appendExchange(
    final String assistantText,
    final List<ToolCall> calls,
    final List<String> results
  ) {
    ToolExchange.requirePaired(calls, results);
    List<ChatMessage> messages = new ArrayList<>(results.size() + 1);
    messages.add(ChatMessage.assistant(ToolMarkup.throughLastClose(
      ToolMarkup.withoutTurnEnds(assistantText), CALL_OPEN, CALL_CLOSE)));
    for (String result : results) {
      messages.add(ChatMessage.tool(RESPONSE_OPEN + result + RESPONSE_CLOSE));
    }
    return messages;
  }

  private record ValueRead(Object value, int next) {
  }
}
