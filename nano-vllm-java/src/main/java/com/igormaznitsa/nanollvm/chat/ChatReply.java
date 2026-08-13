package com.igormaznitsa.nanollvm.chat;

import com.igormaznitsa.nanollvm.llm.GenerationStats;

/**
 * Parsed assistant turn: optional thinking scratchpad, visible answer, and engine stats.
 *
 * <p>Some chat models emit a tagged scratchpad {@code <think>…</think>} before the user-visible
 * reply. {@link #parse(String)} splits decoded assistant text into those two channels and strips
 * chat specials ({@code <|im_end|>}, {@code <end_of_turn>}, …). {@link ChatSession#send} returns a
 * finished turn ({@link #thinkOpen()} is {@code false} and {@link #stats()} is filled). Streaming
 * listeners see partial snapshots: {@code thinkOpen} stays {@code true} until {@code </think>}
 * arrives, and {@code stats} stays {@link GenerationStats#NONE} until the generate completes.
 *
 * <h2 id="typical-use">Typical use after {@code send}</h2>
 * <pre>{@code
 * ChatReply reply = llm.chat(256).send("What is 2+2?");
 * String visible = reply.answer();   // show this to the user; same as reply.text()
 * String notes   = reply.thinking(); // optional UI / logs; not stored in chat history
 * double tokPerSec = reply.stats().completionTokensPerSecond();
 * }</pre>
 *
 * <p>Chat history records {@link #answer()} only — thinking is not replayed on later turns. Null
 * constructor arguments become empty strings / {@link GenerationStats#NONE}. Immutable; safe to
 * share across threads after construction.
 *
 * @param thinking  text inside {@code <think>}…{@code </think>} with markup stripped; empty when
 *                  the model wrote no scratchpad. Several think blocks are joined with newlines.
 * @param answer    visible reply after a closed think block, or the whole decode when there is no
 *                  {@code <think>} tag; chat specials stripped. Empty while thinking is still open,
 *                  or when the model closed the tag and wrote nothing after it.
 * @param thinkOpen {@code true} while an opened {@code <think>} has no matching {@code </think>}
 *                  yet (mid-stream or truncated decode). {@code false} when the scratchpad is
 *                  closed or absent. After {@link ChatSession#send} this is always {@code false}.
 * @param stats     token counts and wall time of the generate that produced this turn.
 *                  {@link GenerationStats#NONE} after plain {@link #parse} and on streaming
 *                  snapshots; session {@code send} attaches measured stats for you.
 */
public record ChatReply(
  String thinking,
  String answer,
  boolean thinkOpen,
  GenerationStats stats
) {

  public ChatReply {
    thinking = thinking == null ? "" : thinking;
    answer = answer == null ? "" : answer;
    stats = stats == null ? GenerationStats.NONE : stats;
  }

  /**
   * Streaming / parse helper without measured stats ({@link GenerationStats#NONE}).
   *
   * @param thinking  scratchpad body, or {@code null} for empty
   * @param answer    visible reply, or {@code null} for empty
   * @param thinkOpen {@code true} if {@code </think>} has not arrived yet
   */
  public ChatReply(final String thinking, final String answer, final boolean thinkOpen) {
    this(thinking, answer, thinkOpen, GenerationStats.NONE);
  }

  static ChatReply from(final AssistantParts parts) {
    return new ChatReply(parts.thinking(), parts.answer(), parts.thinkOpen());
  }

  /**
   * Splits decoded assistant text into thinking / answer / {@code thinkOpen}.
   *
   * <p>Does not salvage a missing answer from the scratchpad and does not attach stats — call
   * {@link #salvageFromThinking(String)} or {@link #withStats(GenerationStats)} yourself, or use
   * {@link ChatSession#send} which does both. A trailing incomplete marker such as {@code <think}
   * is held back so it does not leak into either channel.
   *
   * @param raw decoded assistant tokens (full turn or a streaming prefix); {@code null} / blank
   *            yields empty thinking and answer with {@code thinkOpen == false}
   * @return parsed snapshot; never {@code null}
   */
  public static ChatReply parse(final String raw) {
    return from(AssistantParts.parse(raw));
  }

  /**
   * Visible display string from decoded assistant text: {@link #answer()} when present, otherwise
   * {@link #salvageFromThinking(String)} on the scratchpad.
   *
   * <p>Use when you only need one string for the user and are not driving a split thinking UI.
   *
   * @param raw decoded assistant tokens; {@code null} / blank yields {@code ""}
   * @return stripped visible text, never {@code null}
   */
  public static String cleanAssistantText(final String raw) {
    return AssistantParts.cleanAssistantText(raw);
  }

  /**
   * Incremental display helper for a streaming decode prefix.
   *
   * <p>Same pipeline as {@link #cleanAssistantText(String)}: a closed think block hides the notes
   * and shows the answer; a still-open {@code <think>} surfaces the scratchpad so the UI is not
   * blank while the model is writing notes.
   *
   * @param raw partial decode so far; {@code null} / blank yields {@code ""}
   * @return stripped display text, never {@code null}
   */
  public static String streamDisplayText(final String raw) {
    return AssistantParts.streamDisplayText(raw);
  }

  /**
   * Last-resort visible answer extracted from thinking notes when the answer channel is empty or
   * truncated.
   *
   * <p>Prefers a stated result ({@code answer should be …} / {@code result is …}), then a short
   * last line that does not look like reasoning, then a truncated last line. {@link ChatSession}
   * may call this automatically; {@link #parse(String)} does not.
   *
   * @param thinking scratchpad body; {@code null} / blank yields {@code ""}
   * @return a brief recovered answer, never {@code null}
   */
  public static String salvageFromThinking(final String thinking) {
    return AssistantParts.salvageFromThinking(thinking);
  }

  /**
   * Removes think tags and chat specials from {@code text} without splitting channels.
   *
   * <p>Unlike {@link #parse(String)}, this does not assign content to thinking vs answer — it only
   * strips markup. Useful for sanitizing a string you already treat as a single channel.
   *
   * @param text arbitrary text; {@code null} / empty yields {@code ""}
   * @return {@code text} with chat / think markers and leading {@code assistant:} removed
   */
  public static String stripChatMarkup(final String text) {
    return AssistantParts.stripChatMarkup(text);
  }

  /**
   * Copy of this reply with {@code generateStats} attached (null becomes {@link GenerationStats#NONE}).
   *
   * @param generateStats measured stats from the generate that produced this turn
   * @return a new {@code ChatReply}; this instance is unchanged
   */
  public ChatReply withStats(final GenerationStats generateStats) {
    return new ChatReply(this.thinking, this.answer, this.thinkOpen, generateStats);
  }

  /**
   * Visible answer; same as {@link #answer()}.
   *
   * @return the user-facing reply text (empty while thinking is still open)
   */
  public String text() {
    return this.answer;
  }
}
