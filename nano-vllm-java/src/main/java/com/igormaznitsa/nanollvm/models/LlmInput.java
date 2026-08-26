package com.igormaznitsa.nanollvm.models;

/**
 * Typed payload for {@link LlmModel#generate(LlmInput, LlmModality)} /
 * {@link com.igormaznitsa.nanollvm.llm.LLM#generate(LlmInput, LlmModality)}.
 *
 * <p>In-memory only — load files before constructing an input so {@code generate} stays free of
 * {@link java.io.IOException}.
 *
 * @since 1.3.0
 */
public sealed interface LlmInput permits LlmInText, LlmInSound, LlmInTokenIds {
}
