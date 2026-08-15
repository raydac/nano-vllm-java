/**
 * Chat helpers for library use: typed messages, multi-turn {@link ChatSession}, reply parsing,
 * and unified {@link LlmListener} events ({@link LlmTextKind} text + status).
 *
 * <p>{@link ChatSession} is not thread-safe. Value types ({@link ChatMessage}, {@link ChatReply},
 * {@link ChatHistory}, {@link ThinkTags}, {@link ChatSpecials}, …) are immutable. Prefer {@link ChatReply} for parsed
 * assistant turns ({@code thinking} / {@code answer} / {@code thinkOpen} / {@code stats}). Set
 * scratchpad markers as {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_THINK_TAGS} and
 * answer-search specials as {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_CHAT_SPECIALS}
 * when the checkpoint does not use the library defaults; {@link ChatSession#thinkTags()}
 * overrides the scratchpad pair for one conversation.
 */

package com.igormaznitsa.nanollvm.chat;
