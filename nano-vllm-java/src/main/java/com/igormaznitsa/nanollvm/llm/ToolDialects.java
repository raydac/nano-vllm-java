package com.igormaznitsa.nanollvm.llm;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3_5;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Picks the tool dialect from a checkpoint architecture id
 * ({@link com.igormaznitsa.nanollvm.models.LlmModel#architectureName()}).
 *
 * <p>Qwen3 uses Hermes JSON. Qwen3.5, including 0.8B, uses the XML {@code <function=>} form.
 * LFM2 uses its Pythonic call list. Gemma 4 uses {@code call:} / {@code response:} blocks.
 * Llama and Gemma 3 are rejected: this engine has no Llama 3 tool-turn template, and Gemma 3
 * ships no tool-call tokens.
 *
 * @since 1.5.0
 */
public final class ToolDialects {

  /**
   * How many tool executions one user message may run before the next decode is treated as the
   * reply even if it still contains a call.
   */
  public static final int MAX_EXECUTIONS = ToolExchange.MAX_EXECUTIONS;

  private ToolDialects() {
  }

  /**
   * Dialect for {@code architecture}.
   *
   * @param architecture checkpoint architecture id; must not be {@code null}
   * @return the dialect
   * @throws IllegalArgumentException if this engine cannot round-trip tool calls for that model
   */
  public static ToolDialect of(final String architecture) {
    requireNonNull(architecture, "architecture");
    return switch (architecture) {
      case ARCH_QWEN3 -> new HermesToolDialect();
      case ARCH_QWEN3_5 -> new Qwen35ToolDialect();
      case ARCH_LFM2 -> new Lfm2ToolDialect();
      case ARCH_GEMMA4 -> new Gemma4ToolDialect();
      case ARCH_LLAMA -> throw unsupported(architecture,
        "this engine does not render Llama 3 tool turns");
      case ARCH_GEMMA3 -> throw unsupported(architecture,
        "Gemma 3 has no tool-call tokens; Gemma 4 does");
      default -> throw unsupported(architecture, "no tool-call tokens are wired for it");
    };
  }

  /**
   * Runs {@code calls} against {@code tools}. An unknown name, an unreadable call, or a thrown
   * body becomes an {@code error: ...} string the dialect writes back to the model.
   *
   * @param tools registered tools
   * @param calls parsed calls, in order
   * @return one result per call; never {@code null}
   * @throws NullPointerException if {@code tools} or {@code calls} is {@code null}
   * @since 1.5.0
   */
  public static List<String> execute(final List<ModelTool> tools, final List<ToolCall> calls) {
    return ToolExchange.execute(tools, calls);
  }

  /**
   * Tokenizer stop ids plus any dialect stop whose token is in the vocabulary. The first id stays
   * the engine EOS.
   *
   * @param stopTokenIds ids already chosen for the engine; non-empty
   * @param dialect      dialect selected for the checkpoint
   * @param tokenizer    checkpoint tokenizer
   * @return a new list; the input list is not modified
   */
  static List<Integer> withExtraStops(
    final List<Integer> stopTokenIds,
    final ToolDialect dialect,
    final Tokenizer tokenizer
  ) {
    List<Integer> merged = new ArrayList<>(stopTokenIds);
    for (String token : dialect.extraStopTokens()) {
      tokenizer.tokenId(token).ifPresent(id -> {
        if (!merged.contains(id)) {
          merged.add(id);
        }
      });
    }
    return List.copyOf(merged);
  }

  private static IllegalArgumentException unsupported(final String architecture,
                                                      final String reason) {
    return new IllegalArgumentException(
      "tools are not supported for " + architecture + ": " + reason);
  }
}
