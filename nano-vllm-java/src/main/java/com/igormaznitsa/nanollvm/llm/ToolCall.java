package com.igormaznitsa.nanollvm.llm;

/**
 * One function call parsed from an assistant decode.
 *
 * <p>A blank {@code name} means the model opened a tool-call span the dialect could not read.
 * Whitespace inside the name is removed, so a split token such as {@code os_ name} still matches
 * {@code os_name}. {@code argumentsJson} is a JSON object. The chat loop does not show this
 * record to the user.
 *
 * @param name          function name, or blank when the span was unreadable
 * @param argumentsJson JSON object text; {@code {}} when there are no arguments
 * @since 1.5.0
 */
public record ToolCall(String name, String argumentsJson) {

  /**
   * Canonical constructor. A null name becomes {@code ""}. Whitespace inside the name is removed
   * so a split token still matches the registered tool. A null or blank arguments string becomes
   * {@code {}}.
   */
  public ToolCall {
    name = name == null ? "" : name.replaceAll("\\s+", "");
    argumentsJson = argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson;
  }
}
