package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * Helpers for building and trimming chat histories.
 *
 * <p>{@link #newConversation(String)} returns an unmodifiable list. Truncate / scrub helpers
 * mutate the caller-supplied list in place. {@link ChatSession} calls these before generate so
 * the chat template fits {@code maxModelLen - maxTokens} and filler assistant turns can be
 * stripped on retry.
 */
public final class ChatMessages {

  private static final int PROMPT_MARGIN = 16;

  private ChatMessages() {
  }

  /**
   * Fresh history seeded with an optional system turn.
   *
   * @param systemPrompt instruction text; {@code null} / blank → empty list
   * @return an unmodifiable list (empty when {@code systemPrompt} is null/blank)
   */
  public static List<ChatMessage> newConversation(final String systemPrompt) {
    if (systemPrompt == null || systemPrompt.isBlank()) {
      return List.of();
    }
    return List.of(ChatMessage.system(systemPrompt));
  }

  /**
   * Chat-template maps ({@code role} / {@code content}) in conversation order.
   *
   * @param history turns; must not be {@code null}
   * @return unmodifiable list of two-entry maps
   */
  public static List<Map<String, String>> toTemplateMaps(final List<ChatMessage> history) {
    return history.stream().map(ChatMessage::toMap).toList();
  }

  /**
   * Drops oldest turns from {@code history} until the chat template fits the token budget.
   * Mutates {@code history} in place. Uses {@link Tokenizer#invitesThinking()} and
   * {@link ThinkTags#DEFAULT} for skip-seed accounting.
   *
   * @param history     live session history
   * @param tokenizer   tokenizer that applies the chat template and encodes
   * @param maxModelLen engine context length
   * @param maxTokens   reserved completion budget
   */
  public static void truncateHistory(
    final List<ChatMessage> history,
    final Tokenizer tokenizer,
    final int maxModelLen,
    final int maxTokens
  ) {
    truncateHistory(
      history,
      tokenizer,
      maxModelLen,
      maxTokens,
      tokenizer.invitesThinking(),
      ThinkTags.DEFAULT);
  }

  /**
   * {@link #truncateHistory(List, Tokenizer, int, int)} using the same thinking flag and scratchpad
   * markers as the upcoming generate, so skip-seed tokens count toward the budget.
   *
   * <p>Keeps a leading system turn when present. Drops the oldest non-system turn; if that leaves
   * a dangling assistant message, that is dropped too. Stops at one (or system+one) remaining
   * turn even if still over budget.
   *
   * @param history        live session history
   * @param tokenizer      tokenizer that applies the chat template and encodes
   * @param maxModelLen    engine context length
   * @param maxTokens      reserved completion budget
   * @param enableThinking ChatML skip-seed / thinking invitation for this generate
   * @param thinkTags      scratchpad pair used in the template
   * @throws NullPointerException if {@code thinkTags} is {@code null}
   * @since 1.1.0
   */
  public static void truncateHistory(
    final List<ChatMessage> history,
    final Tokenizer tokenizer,
    final int maxModelLen,
    final int maxTokens,
    final boolean enableThinking,
    final ThinkTags thinkTags
  ) {
    requireNonNull(thinkTags, "thinkTags");
    int budget = Math.max(64, maxModelLen - maxTokens - PROMPT_MARGIN);
    int minKeep = !history.isEmpty() && history.getFirst().role() == ChatRole.SYSTEM ? 2 : 1;
    while (history.size() > minKeep) {
      String prompt = tokenizer.applyChatTemplate(
        toTemplateMaps(history),
        true,
        enableThinking,
        thinkTags.open(),
        thinkTags.close());
      if (tokenizer.encode(prompt).size() <= budget) {
        return;
      }
      int dropAt = history.getFirst().role() == ChatRole.SYSTEM ? 1 : 0;
      if (dropAt >= history.size()) {
        return;
      }
      history.remove(dropAt);
      if (dropAt < history.size() && history.get(dropAt).role() == ChatRole.ASSISTANT) {
        history.remove(dropAt);
      }
    }
  }

  /**
   * Replaces assistant turns matching {@code match} with a short greeting. Mutates {@code history}
   * in place. Used by {@link ChatSession} when retrying after an unusable answer so the model does
   * not latch onto filler history.
   *
   * @param history live session history
   * @param match   predicate on assistant {@link ChatMessage#content()}
   * @since 1.1.0
   */
  public static void scrubMatchingAssistantTurns(
    final List<ChatMessage> history,
    final Predicate<String> match
  ) {
    IntStream.range(0, history.size())
      .filter(i -> history.get(i).role() == ChatRole.ASSISTANT
        && match.test(history.get(i).content()))
      .forEach(i -> history.set(i, ChatMessage.assistant("Hello!")));
  }
}
