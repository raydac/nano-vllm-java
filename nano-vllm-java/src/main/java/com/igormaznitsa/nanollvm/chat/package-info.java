/**
 * Chat helpers for library use: typed messages, multi-turn {@link ChatSession}, reply parsing,
 * and unified {@link LlmListener} events ({@link LlmTextKind} text + status).
 *
 * <p>{@link ChatSession} is not thread-safe. Value types ({@link ChatMessage}, {@link ChatReply},
 * {@link ChatHistory}, …) are immutable. Prefer {@link ChatReply} for parsed assistant turns
 * ({@code thinking} / {@code answer} / {@code thinkOpen} / {@code stats}).
 */

package com.igormaznitsa.nanollvm.chat;
