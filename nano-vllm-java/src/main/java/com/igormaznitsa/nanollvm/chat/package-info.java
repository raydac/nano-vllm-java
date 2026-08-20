/**
 * Chat helpers for library use: typed messages, multi-turn {@link ChatSession}, reply parsing,
 * and unified {@link LlmListener} events ({@link LlmTextKind} text + status).
 *
 * <p>{@link ChatSession} is not thread-safe. Prefer
 * {@link com.igormaznitsa.nanollvm.llm.LLM#chat()} over calling
 * {@code new ChatSession} yourself. Value types ({@link ChatMessage}, {@link ChatReply},
 * {@link ChatHistory}, {@link ThinkTags}, {@link ChatSpecials}, …) are immutable. Show
 * {@link ChatReply#answer()} to the user; {@link ChatReply#thinking()} is an optional private
 * scratchpad some models write before the visible reply. Set scratchpad markers as
 * {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_THINK_TAGS} and leftover chat-markup
 * strings as {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_CHAT_SPECIALS} when the
 * checkpoint does not use the library defaults; {@link ChatSession#thinkTags(ThinkTags)}
 * overrides the scratchpad pair for one conversation.
 *
 * <p><b>Which call.</b> {@link ChatSession#send(String)} is a normal turn (history and generate
 * share the user text). {@link ChatSession#sendPrepared} records one string in history and
 * generates from another (RAG). {@link ChatMessages} trims / scrubs the live history list.
 *
 * <p><b>Events.</b> {@link LlmListener#onText} receives {@link LlmTextEvent}s. Compose sinks with
 * {@link LlmListeners}. CLI tools use {@link LlmListeners#toPrintStreams} ({@link StreamPrinter}).
 */

package com.igormaznitsa.nanollvm.chat;
