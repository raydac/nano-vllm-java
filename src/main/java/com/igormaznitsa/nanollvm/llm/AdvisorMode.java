package com.igormaznitsa.nanollvm.llm;

/**
 * How {@link LLM#setAdvisors(AdvisorMode, String...)} runs advisor generates on the same
 * {@link LLM} before the main chat turn.
 *
 * <ul>
 *   <li>{@link #SEQUENTIAL} — one {@code generate} per advisor prompt</li>
 *   <li>{@link #PARALLEL} — one batched {@code generate} for all advisor prompts</li>
 * </ul>
 */
public enum AdvisorMode {
  SEQUENTIAL,
  PARALLEL
}
