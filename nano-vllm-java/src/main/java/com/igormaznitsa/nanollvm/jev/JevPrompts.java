package com.igormaznitsa.nanollvm.jev;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;
import java.util.Map;

/**
 * Decision prompt: a fixed system line plus a state, a question, and the options in one order.
 */
final class JevPrompts {

  static final String SYSTEM = """
    You are a decision function. You will be given a state and one question. \
    Reply with the answer label only: no words, no punctuation, no explanation.""";

  private JevPrompts() {
  }

  static String userText(
    final String state,
    final JevQuestion question,
    final int[] permutation,
    final List<String> labels
  ) {
    String shown = state == null || state.isBlank() ? "(empty)" : state.strip();
    StringBuilder body = new StringBuilder();
    body.append("State:\n").append(shown).append("\n\n");
    body.append("Question: ").append(question.text()).append('\n');
    if (question.kind() == JevQuestion.Kind.YES_NO) {
      body.append("Answer ")
        .append(labels.get(permutation[0]))
        .append(" or ")
        .append(labels.get(permutation[1]))
        .append('.');
      return body.toString();
    }
    if (question.kind() == JevQuestion.Kind.SCORE) {
      if (question.hasExplicitLevels()) {
        body.append("Pick the level that applies (the levels are ordered):\n");
      } else {
        body.append("Answer on a scale from ")
          .append(compact(question.scaleLow()))
          .append(" to ")
          .append(compact(question.scaleHigh()))
          .append(" by picking the closest bin:\n");
      }
    } else {
      body.append("Options:\n");
    }
    List<String> options = question.options();
    for (int position = 0; position < permutation.length; position++) {
      body.append(labels.get(position))
        .append(". ")
        .append(options.get(permutation[position]))
        .append('\n');
    }
    boolean numeric = !labels.isEmpty() && Character.isDigit(labels.getFirst().charAt(0));
    body.append(numeric ? "Answer with the number only." : "Answer with the letter only.");
    return body.toString();
  }

  static List<Integer> promptTokenIds(
    final Tokenizer tokenizer,
    final ThinkTags thinkTags,
    final String user
  ) {
    requireNonNull(tokenizer, "tokenizer");
    requireNonNull(thinkTags, "thinkTags");
    String rendered = tokenizer.applyChatTemplate(
      List.of(
        Map.of("role", "system", "content", SYSTEM),
        Map.of("role", "user", "content", user)),
      true,
      false,
      thinkTags.open(),
      thinkTags.close());
    List<Integer> ids = tokenizer.encode(rendered);
    if (ids.isEmpty()) {
      throw new IllegalArgumentException("decision prompt encoded to no tokens");
    }
    return ids;
  }

  private static String compact(final double value) {
    if (Double.isFinite(value) && value == Math.rint(value) && Math.abs(value) < 1e15d) {
      return Long.toString(Math.round(value));
    }
    return Double.toString(value);
  }
}
