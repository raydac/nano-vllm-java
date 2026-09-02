package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.fasttext.FastTextModel;
import java.util.List;

/**
 * Meta fastText supervised classifier wrapping {@link FastTextModel}.
 *
 * @since 1.4.0
 */
public final class FastTextForClassification implements TextClassifier {

  public static final String ARCH_FASTTEXT = "fasttext";

  private final FastTextModel model;

  public FastTextForClassification(final FastTextModel model) {
    this.model = requireNonNull(model, "model");
  }

  @Override
  public String architectureName() {
    return ARCH_FASTTEXT;
  }

  @Override
  public List<ScoredLabel> classify(final CharSequence text, final int topK,
                                    final float threshold) {
    requireNonNull(text, "text");
    return this.model.predict(text, topK, threshold).stream()
      .map(prediction -> new ScoredLabel(prediction.label(), prediction.probability()))
      .toList();
  }

  public FastTextModel model() {
    return this.model;
  }
}
