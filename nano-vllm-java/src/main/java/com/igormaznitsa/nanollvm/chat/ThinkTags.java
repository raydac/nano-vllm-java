package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

/**
 * Open/close markers for a tagged thinking scratchpad in decoded assistant text.
 *
 * <p>Not a universal standard: Qwen3 / DeepSeek-R1-style models commonly use
 * {@code <think>}…{@code </think>} ({@link #DEFAULT}). Other families use different strings or
 * none. Pass custom markers at load as {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_THINK_TAGS}
 * so every engine sharing the checkpoint uses the same pair. Override one conversation with
 * {@link ChatSession#thinkTags(ThinkTags)}. {@link ChatReply#parse(String, ThinkTags)} uses the
 * same pair when parsing raw generate output.
 *
 * <pre>{@code
 * LlmModel model = LlmModelFactory.make(path, Map.of(
 *     LlmModel.OPTION_THINK_TAGS, ThinkTags.of("<reasoning>", "</reasoning>")));
 * ChatReply reply = LLM.builder(model).build().chat(256).send("What is 2+2?");
 * }</pre>
 *
 * @param open  start marker; non-blank; must not equal or contain {@code close}
 * @param close end marker; non-blank; must not equal or contain {@code open}
 * @since 1.1.0
 */
public record ThinkTags(String open, String close) {

  public static final ThinkTags DEFAULT = new ThinkTags("<think>", "</think>");

  public ThinkTags {
    requireNonNull(open, "open");
    requireNonNull(close, "close");
    open = open.strip();
    close = close.strip();
    if (open.isEmpty() || close.isEmpty()) {
      throw new IllegalArgumentException("think tags must not be blank");
    }
    if (open.equals(close)) {
      throw new IllegalArgumentException("open and close think tags must differ");
    }
    if (open.contains(close) || close.contains(open)) {
      throw new IllegalArgumentException("think tags must not contain each other");
    }
  }

  /**
   * Custom scratchpad markers (stripped).
   *
   * @param open  start marker; non-blank
   * @param close end marker; non-blank and distinct from {@code open}
   */
  public static ThinkTags of(final String open, final String close) {
    return new ThinkTags(open, close);
  }

  /**
   * Empty scratchpad inserted after the ChatML assistant opener when thinking is disabled and both
   * markers exist in the tokenizer vocab.
   */
  public String skipSeed() {
    return this.open + "\n\n" + this.close + "\n\n";
  }
}
