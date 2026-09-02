package com.igormaznitsa.nanollvm.models.internal;

import java.util.List;

/**
 * Text classification graph (e.g. Meta fastText language-id).
 *
 * @since 1.4.0
 */
public interface TextClassifier {

  /**
   * Architecture id stored on the loaded model (e.g. {@code fasttext}).
   *
   * @return non-blank family key
   */
  String architectureName();

  /**
   * Top labels for {@code text}, filtered by {@code threshold}.
   *
   * @param text      input text; must not be {@code null}
   * @param topK      max labels ({@code >= 1}, or {@code -1} for all)
   * @param threshold minimum score in {@code [0, 1]}
   * @return scored labels, highest first; never {@code null}
   */
  List<ScoredLabel> classify(CharSequence text, int topK, float threshold);

  /**
   * One classification label with score.
   *
   * @param label full label string
   * @param score probability or sigmoid score in {@code [0, 1]}
   * @since 1.4.0
   */
  record ScoredLabel(String label, float score) {
  }
}
