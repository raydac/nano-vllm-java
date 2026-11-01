package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;

/**
 * How one model family advertises tools, emits a call, and reads the result back.
 *
 * <p>Selected from the checkpoint architecture by {@link ToolDialects#of(String)}. Chat uses it
 * for one user turn: render the prompt, parse the decode, run the tools, append the result in
 * this dialect's markup, and generate again.
 *
 * @since 1.5.0
 */
public interface ToolDialect {

  /**
   * Stable id ({@code hermes}, {@code qwen3.5}, {@code lfm2}, {@code gemma4}).
   *
   * @return dialect id
   */
  String id();

  /**
   * Vocabulary strings that must end a generate once tools are registered, in addition to the
   * checkpoint EOS. Empty when the family's normal end-of-turn token already closes a call.
   *
   * @return stop strings; never {@code null}
   */
  List<String> extraStopTokens();

  /**
   * Chat prompt for the next generate, including the tool advertisement and an open assistant turn.
   *
   * @param history        turns so far, including earlier tool results from this user message
   * @param tools          registered tools; non-empty
   * @param tokenizer      checkpoint tokenizer
   * @param enableThinking ChatML scratchpad flag
   * @param tags           scratchpad markers
   * @return prompt text ready to encode
   */
  String render(
    List<ChatMessage> history,
    List<ModelTool> tools,
    Tokenizer tokenizer,
    boolean enableThinking,
    ThinkTags tags
  );

  /**
   * Complete tool calls in {@code assistantText}, in emission order. An unreadable span is a
   * {@link ToolCall} with a blank name so the model receives an error result.
   *
   * @param assistantText raw decode, special tokens kept
   * @return calls; empty when the model did not open a complete call
   */
  List<ToolCall> parseCalls(String assistantText);

  /**
   * Assistant text with tool-call spans removed, for the user-visible channel.
   *
   * @param assistantText raw decode
   * @return prose outside call spans
   */
  String visibleText(String assistantText);

  /**
   * History rows to append after {@code calls} have been executed: the assistant call, then the
   * result in this family's markup. Text the model wrote after the last call is omitted, so the
   * next generate can still request another tool. Gemma 4 keeps the result inside the same
   * assistant turn.
   *
   * @param assistantText raw decode that contained {@code calls}
   * @param calls         parsed calls, same order as {@code results}
   * @param results       tool bodies, one per call
   * @return messages to append before the next generate
   */
  List<ChatMessage> appendExchange(
    String assistantText,
    List<ToolCall> calls,
    List<String> results
  );
}
