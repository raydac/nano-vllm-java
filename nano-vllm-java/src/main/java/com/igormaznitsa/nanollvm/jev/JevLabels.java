package com.igormaznitsa.nanollvm.jev;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.HashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Maps answer labels to the single vocabulary ids whose logits are read.
 */
final class JevLabels {

  private JevLabels() {
  }

  static Resolved resolve(final Tokenizer tokenizer, final JevQuestion question) {
    try {
      return bind(tokenizer, question.answerLabels());
    } catch (LabelTokenException failure) {
      if (question.kind() != JevQuestion.Kind.SCORE) {
        throw failure;
      }
      return bind(tokenizer, question.letterLabels());
    }
  }

  static int[] idsForRotation(final JevQuestion question, final int[] labelIds,
                              final int[] permutation) {
    if (question.kind() != JevQuestion.Kind.YES_NO) {
      return labelIds;
    }
    int[] positioned = new int[permutation.length];
    for (int position = 0; position < permutation.length; position++) {
      positioned[position] = labelIds[permutation[position]];
    }
    return positioned;
  }

  private static Resolved bind(final Tokenizer tokenizer, final List<String> labels) {
    int[] ids = new int[labels.size()];
    for (int index = 0; index < labels.size(); index++) {
      ids[index] = singleToken(tokenizer, labels.get(index));
    }
    Set<Integer> seen = new HashSet<>();
    for (int index = 0; index < ids.length; index++) {
      if (!seen.add(ids[index])) {
        throw new LabelTokenException("label tokens collide for \"" + labels.get(index) + "\"");
      }
    }
    return new Resolved(List.copyOf(labels), ids);
  }

  private static int singleToken(final Tokenizer tokenizer, final String label) {
    OptionalInt bare = oneId(tokenizer, label);
    if (bare.isPresent() && roundTrips(tokenizer, bare.getAsInt(), label)) {
      return bare.getAsInt();
    }
    OptionalInt spaced = oneId(tokenizer, " " + label);
    if (spaced.isPresent() && roundTrips(tokenizer, spaced.getAsInt(), label)) {
      return spaced.getAsInt();
    }
    throw new LabelTokenException(
      "label \"" + label + "\" is not a single token for this tokenizer");
  }

  private static boolean roundTrips(final Tokenizer tokenizer, final int id, final String label) {
    String decoded = tokenizer.decode(List.of(id), false);
    return label.equals(decoded) || label.equals(decoded.strip()) || (" " + label).equals(decoded);
  }

  private static OptionalInt oneId(final Tokenizer tokenizer, final String text) {
    List<Integer> ids = tokenizer.encode(text);
    if (ids.size() == 1) {
      return OptionalInt.of(ids.getFirst());
    }
    return OptionalInt.empty();
  }

  record Resolved(List<String> labels, int[] ids) {
  }

  private static final class LabelTokenException extends IllegalArgumentException {

    private LabelTokenException(final String message) {
      super(message);
    }
  }
}
