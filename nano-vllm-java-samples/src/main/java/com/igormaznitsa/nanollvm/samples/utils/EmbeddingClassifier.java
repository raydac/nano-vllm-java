package com.igormaznitsa.nanollvm.samples.utils;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Few-shot label probe on frozen encoder vectors ({@code LlmModel.embed}).
 *
 * <p>Fits one L2-normalized residual prototype per label after subtracting the training mean.
 * That centering is what makes fill-mask XLM-RoBERTa / BERT vectors usable; raw cosine is not.
 * This is not a Hub sequence-classification head and does not train the encoder.
 *
 * @since 1.3.0
 */
public final class EmbeddingClassifier {

  private final List<String> labels;
  private final float[] trainMean;
  private final float[][] prototypes;
  private final int exampleCount;

  private EmbeddingClassifier(
    final List<String> labels,
    final float[] trainMean,
    final float[][] prototypes,
    final int exampleCount
  ) {
    this.labels = List.copyOf(labels);
    this.trainMean = trainMean;
    this.prototypes = prototypes;
    this.exampleCount = exampleCount;
  }

  public static Trainer trainer() {
    return new Trainer();
  }

  /**
   * Splits {@code label | text} or {@code label<TAB>text}. Blank lines and {@code #} comments
   * are absent.
   */
  public static Optional<LabeledText> parseLabeledLine(final String line) {
    requireNonNull(line, "line");
    String stripped = line.strip();
    if (stripped.isEmpty() || stripped.charAt(0) == '#') {
      return Optional.empty();
    }

    int tab = stripped.indexOf('\t');
    int pipe = stripped.indexOf('|');
    int split = splitIndex(tab, pipe);
    if (split < 0) {
      return Optional.empty();
    }

    String label = stripped.substring(0, split).strip();
    String text = stripped.substring(split + 1).strip();
    if (label.isEmpty() || text.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(new LabeledText(normalizeLabel(label), text));
  }

  private static int splitIndex(final int tab, final int pipe) {
    if (tab >= 0 && (pipe < 0 || tab < pipe)) {
      return tab;
    }
    return pipe;
  }

  static String normalizeLabel(final String label) {
    return requireNonNull(label, "label").strip().toLowerCase(Locale.ROOT);
  }

  private static float[] requireVector(final float[] embedding) {
    requireNonNull(embedding, "embedding");
    if (embedding.length == 0) {
      throw new IllegalArgumentException("embedding must not be empty");
    }
    return embedding.clone();
  }

  private static float[] meanOf(final List<float[]> vectors, final int dimensions) {
    float[] mean = new float[dimensions];
    for (float[] vector : vectors) {
      for (int d = 0; d < dimensions; d++) {
        mean[d] += vector[d];
      }
    }
    float inv = 1f / vectors.size();
    for (int d = 0; d < dimensions; d++) {
      mean[d] *= inv;
    }
    return mean;
  }

  private static float[] residualCentroid(
    final List<float[]> vectors,
    final float[] mean,
    final int dimensions
  ) {
    float[] sum = new float[dimensions];
    for (float[] vector : vectors) {
      for (int d = 0; d < dimensions; d++) {
        sum[d] += vector[d] - mean[d];
      }
    }
    float inv = 1f / vectors.size();
    for (int d = 0; d < dimensions; d++) {
      sum[d] *= inv;
    }
    l2NormalizeInPlace(sum);
    return sum;
  }

  private static float[] centeredUnit(final float[] embedding, final float[] mean) {
    float[] centered = new float[embedding.length];
    for (int d = 0; d < embedding.length; d++) {
      centered[d] = embedding[d] - mean[d];
    }
    l2NormalizeInPlace(centered);
    return centered;
  }

  private static void l2NormalizeInPlace(final float[] vector) {
    double sumSq = 0.0;
    for (float value : vector) {
      sumSq += (double) value * value;
    }
    if (sumSq <= 0.0) {
      return;
    }
    float inv = (float) (1.0 / Math.sqrt(sumSq));
    for (int i = 0; i < vector.length; i++) {
      vector[i] *= inv;
    }
  }

  private static double dot(final float[] left, final float[] right) {
    double sum = 0.0;
    for (int i = 0; i < left.length; i++) {
      sum += (double) left[i] * right[i];
    }
    return sum;
  }

  public List<String> labels() {
    return this.labels;
  }

  public int exampleCount() {
    return this.exampleCount;
  }

  public int dimensions() {
    return this.trainMean.length;
  }

  public Prediction classify(final float[] embedding) {
    float[] copy = requireVector(embedding);
    if (copy.length != this.trainMean.length) {
      throw new IllegalArgumentException(
        "embedding length %d must match training dim %d"
          .formatted(copy.length, this.trainMean.length));
    }
    float[] query = centeredUnit(copy, this.trainMean);

    List<ClassScore> scores = IntStream.range(0, this.labels.size())
      .mapToObj(i -> new ClassScore(this.labels.get(i), dot(query, this.prototypes[i])))
      .sorted(Comparator.comparingDouble(ClassScore::cosine).reversed()
        .thenComparing(ClassScore::label))
      .toList();
    return new Prediction(scores.getFirst().label(), scores);
  }

  public record LabeledText(String label, String text) {
    public LabeledText {
      label = normalizeLabel(label);
      requireNonNull(text, "text");
      if (label.isEmpty()) {
        throw new IllegalArgumentException("label must not be blank");
      }
      if (text.isBlank()) {
        throw new IllegalArgumentException("text must not be blank");
      }
    }
  }

  public record ClassScore(String label, double cosine) {
    public ClassScore {
      requireNonNull(label, "label");
    }
  }

  public record Prediction(String label, List<ClassScore> scores) {
    public Prediction {
      requireNonNull(label, "label");
      scores = List.copyOf(requireNonNull(scores, "scores"));
      if (scores.isEmpty()) {
        throw new IllegalArgumentException("scores must not be empty");
      }
    }

    public boolean isClose(final double margin) {
      if (this.scores.size() < 2) {
        return false;
      }
      return this.scores.getFirst().cosine() - this.scores.get(1).cosine() < margin;
    }
  }

  public static final class Trainer {

    private final List<HeldExample> examples = new ArrayList<>();

    public Trainer add(final String label, final float[] embedding) {
      String normalized = normalizeLabel(label);
      if (normalized.isEmpty()) {
        throw new IllegalArgumentException("label must not be blank");
      }
      float[] copy = requireVector(embedding);
      if (!this.examples.isEmpty() && copy.length != this.examples.getFirst().embedding().length) {
        throw new IllegalArgumentException(
          "embedding length %d must match prior examples (%d)"
            .formatted(copy.length, this.examples.getFirst().embedding().length));
      }
      this.examples.add(new HeldExample(normalized, copy));
      return this;
    }

    public Trainer clear() {
      this.examples.clear();
      return this;
    }

    public int size() {
      return this.examples.size();
    }

    public List<String> labels() {
      return this.examples.stream()
        .map(HeldExample::label)
        .distinct()
        .toList();
    }

    public boolean canFit() {
      return this.labels().size() >= 2;
    }

    public EmbeddingClassifier fit() {
      if (!this.canFit()) {
        throw new IllegalStateException("need examples for at least two labels");
      }

      int dimensions = this.examples.getFirst().embedding().length;
      List<float[]> vectors = this.examples.stream().map(HeldExample::embedding).toList();
      float[] mean = meanOf(vectors, dimensions);

      Map<String, List<float[]>> byLabel = new LinkedHashMap<>();
      this.examples.forEach(example ->
        byLabel.computeIfAbsent(example.label(), key -> new ArrayList<>())
          .add(example.embedding()));

      List<String> labels = List.copyOf(byLabel.keySet());
      float[][] prototypes = labels.stream()
        .map(label -> residualCentroid(byLabel.get(label), mean, dimensions))
        .toArray(float[][]::new);
      return new EmbeddingClassifier(labels, mean, prototypes, this.examples.size());
    }

    @SuppressWarnings("ArrayRecordComponent")
    private record HeldExample(String label, float[] embedding) {
    }
  }
}
