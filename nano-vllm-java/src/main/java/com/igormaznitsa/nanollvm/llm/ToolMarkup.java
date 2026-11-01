package com.igormaznitsa.nanollvm.llm;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared span search for tool-call markup. Dialects own the open and close strings.
 */
final class ToolMarkup {

  private static final List<String> TURN_ENDS = List.of(
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|tool_response>",
    "<turn|>",
    "<end_of_turn>"
  );

  private ToolMarkup() {
  }

  /**
   * Bodies of complete {@code open ... close} spans, in order. An unclosed tail is ignored.
   *
   * @param text  decode or prompt text; never {@code null}
   * @param open  opening marker
   * @param close closing marker
   * @return bodies without the markers; empty when {@code text} has no complete span
   */
  static List<String> bodies(final String text, final String open, final String close) {
    List<String> bodies = new ArrayList<>();
    int cursor = 0;
    while (cursor < text.length()) {
      int start = text.indexOf(open, cursor);
      if (start < 0) {
        break;
      }
      int bodyStart = start + open.length();
      int end = text.indexOf(close, bodyStart);
      if (end < 0) {
        break;
      }
      bodies.add(text.substring(bodyStart, end));
      cursor = end + close.length();
    }
    return bodies;
  }

  /**
   * Text up to and including the last complete call. Prose after that call is dropped so the next
   * turn still asks for a tool instead of continuing an answer. Text with no complete call is kept.
   *
   * @param text  assistant decode; {@code null} or empty yields {@code ""}
   * @param open  opening marker
   * @param close closing marker
   * @return stripped text through the last close, or the stripped original when no call closes
   */
  static String throughLastClose(final String text, final String open, final String close) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    int cursor = 0;
    int lastEnd = -1;
    while (cursor < text.length()) {
      int start = text.indexOf(open, cursor);
      if (start < 0) {
        break;
      }
      int end = text.indexOf(close, start + open.length());
      if (end < 0) {
        break;
      }
      lastEnd = end + close.length();
      cursor = lastEnd;
    }
    return (lastEnd < 0 ? text : text.substring(0, lastEnd)).strip();
  }

  /**
   * Drops complete call spans and any unfinished open span at the tail. Prose before and between
   * calls stays, which is the user-visible channel.
   *
   * @param text  assistant decode; {@code null} or empty yields {@code ""}
   * @param open  opening marker
   * @param close closing marker
   * @return text outside call spans
   */
  static String hide(final String text, final String open, final String close) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    StringBuilder kept = new StringBuilder(text.length());
    int cursor = 0;
    while (cursor < text.length()) {
      int start = text.indexOf(open, cursor);
      if (start < 0) {
        kept.append(text, cursor, text.length());
        break;
      }
      kept.append(text, cursor, start);
      int end = text.indexOf(close, start + open.length());
      if (end < 0) {
        break;
      }
      cursor = end + close.length();
    }
    return kept.toString();
  }

  /**
   * Removes trailing end-of-turn and tool-response opener tokens so a stored assistant turn does
   * not embed the stop that ended the generate. A marker is removed only when it is a suffix, so
   * {@code <|im_end|>} inside a longer token is left alone.
   *
   * @param text assistant decode; {@code null} is treated as empty
   * @return text with those suffixes removed
   */
  static String withoutTurnEnds(final String text) {
    String stripped = text == null ? "" : text.strip();
    boolean removed = true;
    while (removed && !stripped.isEmpty()) {
      removed = false;
      for (String end : TURN_ENDS) {
        if (stripped.endsWith(end)) {
          stripped = stripped.substring(0, stripped.length() - end.length()).stripTrailing();
          removed = true;
        }
      }
    }
    return stripped;
  }
}
