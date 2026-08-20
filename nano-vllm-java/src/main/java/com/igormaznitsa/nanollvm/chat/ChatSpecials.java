package com.igormaznitsa.nanollvm.chat;

import static java.util.Comparator.comparingInt;
import static java.util.Comparator.naturalOrder;
import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Special-token strings searched in decoded assistant text so chat markup can be stripped from the
 * visible answer.
 *
 * <p>Not a universal vocabulary: ChatML, Gemma turn markers, Llama stops, and default
 * {@link ThinkTags} are listed in {@link #DEFAULT}. Pass a custom list at load as
 * {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_CHAT_SPECIALS} so every engine sharing
 * the checkpoint uses the same search set. {@link ChatReply#parse(String, ThinkTags, ChatSpecials)}
 * uses that list (plus the active {@link ThinkTags} pair) when splitting a generate.
 *
 * <pre>{@code
 * LlmModel model = LlmModelFactory.open(path)
 *     .chatSpecials(ChatSpecials.of("<|im_end|>", "<|eot_id|>"))
 *     .make();
 * }</pre>
 *
 * @param markers distinct non-blank search strings, longest first; empty means strip no chat
 *                specials (think tags are still merged at parse)
 * @since 1.1.0
 */
public record ChatSpecials(List<String> markers) {

  /**
   * ChatML, Gemma turn, Llama header/stop, and default {@link ThinkTags} strings, longest first.
   */
  public static final ChatSpecials DEFAULT = new ChatSpecials(List.of(
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<end_of_turn>",
    "<start_of_turn>",
    "<|turn>user\n",
    "<|turn>model\n",
    "<|turn>system\n",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<eos>",
    "<bos>",
    "</s>",
    ThinkTags.DEFAULT.open(),
    ThinkTags.DEFAULT.close()
  ));

  /**
   * Canonical constructor: strips, rejects blank markers, de-duplicates, sorts longest first.
   *
   * @throws NullPointerException     if {@code markers} or an element is {@code null}
   * @throws IllegalArgumentException if a marker is blank after strip
   */
  public ChatSpecials {
    requireNonNull(markers, "markers");
    markers = List.copyOf(longestFirst(markers.stream()
      .map(ChatSpecials::requireNonBlankMarker)
      .distinct()
      .toList()));
  }

  /**
   * Distinct non-blank search strings (sorted longest first at construction).
   *
   * @param markers markers to search; must not be {@code null} (may be empty)
   * @return validated specials
   * @since 1.1.0
   */
  public static ChatSpecials of(final String... markers) {
    return new ChatSpecials(List.of(requireNonNull(markers, "markers")));
  }

  /**
   * Distinct non-blank search strings (sorted longest first at construction).
   *
   * @param markers markers to search; must not be {@code null}
   * @return validated specials
   * @since 1.1.0
   */
  public static ChatSpecials of(final List<String> markers) {
    return new ChatSpecials(requireNonNull(markers, "markers"));
  }

  /**
   * Strips {@code marker}; rejects {@code null} and blank after strip.
   */
  private static String requireNonBlankMarker(final String marker) {
    String stripped = requireNonNull(marker, "marker").strip();
    if (stripped.isEmpty()) {
      throw new IllegalArgumentException("chat special must not be blank");
    }
    return stripped;
  }

  /** Stable sort: longer strings first, then lexicographic. */
  private static List<String> longestFirst(final List<String> markers) {
    return markers.stream()
      .sorted(comparingInt(String::length).reversed().thenComparing(naturalOrder()))
      .toList();
  }

  /**
   * Markers to search in a decode, including {@code tags} when they are not already in this list
   * (longest first). Used when splitting a generate so think tags are stripped even if this list
   * omitted them.
   *
   * @param tags scratchpad pair merged into {@link #markers()} when absent
   * @return unmodifiable list, longest first
   */
  List<String> searchMarkers(final ThinkTags tags) {
    requireNonNull(tags, "tags");
    if (this.markers.contains(tags.open()) && this.markers.contains(tags.close())) {
      return this.markers;
    }

    List<String> merged = new ArrayList<>(this.markers.size() + 2);
    merged.addAll(this.markers);
    if (!merged.contains(tags.open())) {
      merged.add(tags.open());
    }
    if (!merged.contains(tags.close())) {
      merged.add(tags.close());
    }
    return List.copyOf(longestFirst(merged));
  }
}
