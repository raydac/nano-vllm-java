package com.igormaznitsa.nanollvm.models.internal.fasttext;

import static java.util.Objects.requireNonNull;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Meta fastText supervised model (dense or ProductQuantizer) for language-id and similar tasks.
 *
 * @since 1.4.0
 */
public final class FastTextModel {

  public static final int MAGIC = 793712314;
  private static final int MAX_VERSION = 12;
  private static final int LINE_CAPACITY = 1 << 16;
  private static final int UNLIMITED_K = -1;

  private final Args args;
  private final Dictionary dictionary;
  private final Matrix input;
  private final Matrix output;
  private final boolean quantized;
  private final HierarchicalSoftmax hierarchicalSoftmax;

  private FastTextModel(
    final Args args,
    final Dictionary dictionary,
    final Matrix input,
    final Matrix output,
    final boolean quantized,
    final HierarchicalSoftmax hierarchicalSoftmax
  ) {
    this.args = args;
    this.dictionary = dictionary;
    this.input = input;
    this.output = output;
    this.quantized = quantized;
    this.hierarchicalSoftmax = hierarchicalSoftmax;
  }

  public static boolean isFastTextFile(final Path path) {
    try (InputStream raw = Files.newInputStream(path);
         BufferedInputStream buffered = new BufferedInputStream(raw)) {
      final LittleEndianInput in = new LittleEndianInput(buffered);
      return in.readInt() == MAGIC;
    } catch (final IOException ignored) {
      return false;
    }
  }

  public static FastTextModel load(final Path path) throws IOException {
    requireNonNull(path, "path");
    try (InputStream raw = Files.newInputStream(path);
         BufferedInputStream buffered = new BufferedInputStream(raw, 1 << 20)) {
      return load(new LittleEndianInput(buffered));
    }
  }

  private static FastTextModel load(final LittleEndianInput in) throws IOException {
    final int magic = in.readInt();
    if (magic != MAGIC) {
      throw new IOException("Not a fastText model (bad magic)");
    }
    final int version = in.readInt();
    if (version > MAX_VERSION) {
      throw new IOException("Unsupported fastText version: " + version);
    }

    final Args args = Args.load(in, version);
    if (args.model != Args.MODEL_SUP) {
      throw new IOException(
        "Only supervised fastText models are supported (got model=" + args.model + ")");
    }
    if (args.loss != Args.LOSS_SOFTMAX
      && args.loss != Args.LOSS_OVA
      && args.loss != Args.LOSS_HS) {
      throw new IOException(
        "Unsupported fastText loss for predict (need softmax/ova/hs, got " + args.loss + ")");
    }

    final Dictionary dictionary = Dictionary.load(in, args);

    final boolean quantInput = in.readBoolean();
    final Matrix input = quantInput ? QuantMatrix.load(in) : DenseMatrix.load(in);
    if (!quantInput && dictionary.isPruned()) {
      throw new IOException(
        "Invalid model file. Please download the updated model from www.fasttext.cc "
          + "(see facebookresearch/fastText issue #332).");
    }

    args.qout = in.readBoolean();
    final Matrix output =
      quantInput && args.qout ? QuantMatrix.load(in) : DenseMatrix.load(in);

    final HierarchicalSoftmax hierarchical = args.loss == Args.LOSS_HS
      ? HierarchicalSoftmax.build(dictionary.labelCounts(), output)
      : null;
    return new FastTextModel(args, dictionary, input, output, quantInput, hierarchical);
  }

  public int dimension() {
    return this.args.dim;
  }

  public int labelCount() {
    return this.dictionary.nlabels();
  }

  public boolean quantized() {
    return this.quantized;
  }

  public List<Prediction> predict(final CharSequence text, final int k, final float threshold) {
    requireNonNull(text, "text");
    if (k == 0 || k < UNLIMITED_K) {
      throw new IllegalArgumentException("k needs to be 1 or higher (or -1 for all labels)");
    }

    final int[] line = new int[LINE_CAPACITY];
    final int lineSize = this.dictionary.getLine(text, line, LINE_CAPACITY);
    if (lineSize == 0) {
      return List.of();
    }

    final float[] hidden = new float[this.args.dim];
    this.input.averageRowsToVector(hidden, line, lineSize);

    if (this.hierarchicalSoftmax != null) {
      final int topK = k == UNLIMITED_K ? this.dictionary.nlabels() : k;
      return this.hierarchicalSoftmax.predict(hidden, this.dictionary, topK, threshold);
    }

    final int outputSize = (int) this.output.rows();
    final float[] scores = new float[outputSize];
    this.computeOutput(hidden, scores);

    final int topK = k == UNLIMITED_K ? outputSize : k;
    return this.findKBest(scores, topK, threshold);
  }

  private void computeOutput(final float[] hidden, final float[] scores) {
    for (int i = 0; i < scores.length; i++) {
      scores[i] = this.output.dotRow(hidden, i);
    }
    if (this.args.loss == Args.LOSS_OVA) {
      for (int i = 0; i < scores.length; i++) {
        scores[i] = this.sigmoid(scores[i]);
      }
      return;
    }
    this.softmaxInPlace(scores);
  }

  private void softmaxInPlace(final float[] scores) {
    float max = scores[0];
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > max) {
        max = scores[i];
      }
    }
    float z = 0f;
    for (int i = 0; i < scores.length; i++) {
      scores[i] = (float) Math.exp(scores[i] - max);
      z += scores[i];
    }
    final float inv = 1f / z;
    for (int i = 0; i < scores.length; i++) {
      scores[i] *= inv;
    }
  }

  private float sigmoid(final float x) {
    if (x < -8f) {
      return 0f;
    }
    if (x > 8f) {
      return 1f;
    }
    return (float) (1.0 / (1.0 + Math.exp(-x)));
  }

  private List<Prediction> findKBest(final float[] scores, final int k, final float threshold) {
    if (k < 1) {
      return List.of();
    }
    int limit = Math.min(k, scores.length);
    int[] heap = new int[limit];
    int size = 0;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] < threshold) {
        continue;
      }
      if (size == limit && scores[i] <= scores[heap[0]]) {
        continue;
      }
      if (size < limit) {
        heap[size] = i;
        this.siftUpWorst(scores, heap, size);
        size++;
        continue;
      }
      heap[0] = i;
      this.siftDownWorst(scores, heap, size, 0);
    }

    List<Prediction> result = new ArrayList<>(size);
    for (int i = 0; i < size; i++) {
      int labelId = heap[i];
      result.add(new Prediction(this.dictionary.getLabel(labelId), scores[labelId]));
    }
    result.sort(Comparator.comparingDouble(Prediction::probability).reversed());
    return List.copyOf(result);
  }

  private void siftUpWorst(final float[] scores, final int[] heap, final int index) {
    int node = index;
    while (node > 0) {
      int parent = (node - 1) / 2;
      if (scores[heap[node]] >= scores[heap[parent]]) {
        break;
      }
      this.swap(heap, node, parent);
      node = parent;
    }
  }

  private void siftDownWorst(
    final float[] scores, final int[] heap, final int size, final int index
  ) {
    int node = index;
    while (true) {
      int left = node * 2 + 1;
      if (left >= size) {
        return;
      }
      int right = left + 1;
      int worst = left;
      if (right < size && scores[heap[right]] < scores[heap[left]]) {
        worst = right;
      }
      if (scores[heap[worst]] >= scores[heap[node]]) {
        return;
      }
      this.swap(heap, node, worst);
      node = worst;
    }
  }

  private void swap(final int[] values, final int left, final int right) {
    int held = values[left];
    values[left] = values[right];
    values[right] = held;
  }

  /**
   * One supervised label with its probability (or sigmoid score for OVA).
   *
   * @param label       full label string including {@code __label__} prefix
   * @param probability model score in {@code [0, 1]}
   * @since 1.4.0
   */
  public record Prediction(String label, float probability) {
  }
}
