package com.igormaznitsa.nanollvm.llm;

/**
 * How {@link LLM#setSubagents(SubagentMode, String...)} runs advisor generates on the same
 * {@link LLM} before the main chat turn.
 *
 * <ul>
 *   <li>{@link #SEQUENTIAL} — one {@code generate} per subagent prompt</li>
 *   <li>{@link #PARALLEL} — one batched {@code generate} for all subagent prompts</li>
 * </ul>
 */
public enum SubagentMode {
  SEQUENTIAL,
  PARALLEL
}
