package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Json;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * One function the chat model may call during {@link com.igormaznitsa.nanollvm.chat.ChatSession#send}.
 *
 * <p>Register it on the engine with {@link LLM.Builder#tool(ModelTool...)}. The engine advertises
 * the name, description, and JSON Schema in the checkpoint's own tool prompt, parses the model's
 * call, and passes {@code argumentsJson} (a JSON object) to {@link ToolBody#call(String)}. The
 * returned text is written back in that model's tool-result markup, then generation continues.
 *
 * <p>The body runs on the chat thread. Throw to report a failure; the engine sends the exception
 * message back to the model instead of failing the turn. Immutable.
 *
 * @param name             function name the model writes; {@code [A-Za-z_][A-Za-z0-9_]*}
 * @param description      when the model should call this function
 * @param parametersSchema JSON Schema object for the arguments ({@code {"type":"object", ...}})
 * @param body             executes one call
 * @since 1.5.0
 */
public record ModelTool(
  String name,
  String description,
  String parametersSchema,
  ToolBody body
) {

  private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

  /**
   * Canonical constructor. Rejects a blank description, a name the tool dialects cannot emit, and
   * a parameters value that is not a JSON object.
   *
   * @throws NullPointerException     if any component is {@code null}
   * @throws IllegalArgumentException if the name, description, or schema is unusable
   */
  public ModelTool {
    requireNonNull(name, "name");
    requireNonNull(description, "description");
    requireNonNull(parametersSchema, "parametersSchema");
    requireNonNull(body, "body");
    if (!NAME.matcher(name).matches()) {
      throw new IllegalArgumentException(
        "tool name must match [A-Za-z_][A-Za-z0-9_]*, got: " + name);
    }
    if (description.isBlank()) {
      throw new IllegalArgumentException("tool description must not be blank");
    }
    if (!(Json.parse(parametersSchema) instanceof Map<?, ?>)) {
      throw new IllegalArgumentException("tool parametersSchema must be a JSON object");
    }
  }

  /**
   * Registers a function. {@code parametersSchema} is a JSON Schema object.
   *
   * @param name             function name the model writes
   * @param description      when to call it
   * @param parametersSchema JSON Schema object text
   * @param body             executes one call; must not be {@code null}
   * @return the tool
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if the name, description, or schema is unusable
   * @since 1.5.0
   */
  public static ModelTool of(
    final String name,
    final String description,
    final String parametersSchema,
    final ToolBody body
  ) {
    return new ModelTool(name, description, parametersSchema, body);
  }

  /**
   * Runs the body on the caller thread. A null arguments string is passed as {@code {}}.
   * An exception thrown by the body propagates to the caller; the chat loop turns it into an
   * {@code error: ...} result for the model.
   *
   * @param argumentsJson JSON object text from the model
   * @return the body result; {@code ""} when the body returns {@code null}
   * @since 1.5.0
   */
  public String call(final String argumentsJson) {
    String result = this.body.call(argumentsJson == null ? "{}" : argumentsJson);
    return result == null ? "" : result;
  }

  /**
   * Executes one parsed call.
   *
   * @since 1.5.0
   */
  @FunctionalInterface
  public interface ToolBody {

    /**
     * Runs one call. The chat loop invokes this once per parsed call and sends the text back
     * in the checkpoint's tool-result markup.
     *
     * @param argumentsJson JSON object text; never {@code null}. An empty call is {@code {}}
     * @return text shown back to the model; {@code null} is stored as empty
     * @throws RuntimeException to report a failure the model should see as {@code error: ...}
     */
    String call(String argumentsJson);
  }
}
