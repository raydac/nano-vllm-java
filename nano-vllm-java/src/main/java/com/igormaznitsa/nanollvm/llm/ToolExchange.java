package com.igormaznitsa.nanollvm.llm;

import static java.util.stream.Collectors.toMap;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs parsed calls against the tools registered on the engine.
 */
final class ToolExchange {

  static final int MAX_EXECUTIONS = 4;

  private ToolExchange() {
  }

  static void requirePaired(final List<ToolCall> calls, final List<String> results) {
    if (calls.size() != results.size()) {
      throw new IllegalArgumentException(
        "tool results (%d) must match calls (%d)".formatted(results.size(), calls.size()));
    }
  }

  static List<String> execute(final List<ModelTool> tools, final List<ToolCall> calls) {
    Map<String, ModelTool> byName = tools.stream()
      .collect(toMap(ModelTool::name, tool -> tool, (left, right) -> left, LinkedHashMap::new));
    return calls.stream().map(call -> invoke(byName, call)).toList();
  }

  private static String invoke(final Map<String, ModelTool> byName, final ToolCall call) {
    if (call.name().isBlank()) {
      return "error: unreadable tool call";
    }
    ModelTool tool = byName.get(call.name());
    if (tool == null) {
      return "error: unknown tool " + call.name();
    }
    try {
      return tool.call(call.argumentsJson());
    } catch (RuntimeException ex) {
      String message = ex.getMessage();
      return "error: " + (message == null || message.isBlank()
        ? ex.getClass().getSimpleName()
        : message);
    }
  }
}
