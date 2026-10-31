package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Next-token draw from a batch of logits: temperature softmax, then top-k, then top-p, then
 * Gumbel-max. Used by {@link com.igormaznitsa.nanollvm.engine.Transformer} once per generate step.
 *
 * <p>Order matches {@link com.igormaznitsa.nanollvm.llm.SamplingParams}: logits are divided by
 * temperature, softmaxed, top-k zeros the tail and renormalizes, top-p keeps the smallest prefix
 * whose mass is at least {@code p} and renormalizes, then one index is drawn. {@code topK == 0}
 * skips top-k; {@code topP >= 1} skips nucleus.
 *
 * <p>This type holds no RNG. The caller supplies a {@link Random} ({@code LLM} owns one per
 * engine and passes it through {@code Transformer} under the exclusive generate lock).
 * {@code topK == 1} is greedy argmax and does not draw. Temperature must be {@code > 0};
 * {@code SamplingParams} already rejects greedy {@code 0}.
 *
 * <p><strong>Thread safety:</strong> {@link #forward} is safe for concurrent callers only when
 * each call uses a distinct {@link Random}; sharing one {@code Random} across threads is not.
 *
 * @see com.igormaznitsa.nanollvm.llm.SamplingParams
 */
public final class Sampler {

  /**
   * Draws one token id per logit row.
   *
   * @param logits       {@code [rows, vocab]}
   * @param temperatures per-row softmax temperature; length {@code rows}
   * @param topKs        per-row top-k ({@code 0} = disabled), or {@code null} for all disabled
   * @param topPs        per-row nucleus ({@code 1} = disabled), or {@code null} for all {@code 1}
   * @param random       Gumbel draw source; must not be {@code null}
   * @return token ids, length {@code rows}
   */
  public int[] forward(
    final Tensor logits,
    final float[] temperatures,
    final int[] topKs,
    final float[] topPs,
    final Random random
  ) {
    requireNonNull(random, "random");
    int rows = logits.size(0);
    int vocab = logits.size(1);
    int[] out = new int[rows];
    for (int r = 0; r < rows; r++) {
      float temperature = temperatures[r];
      int topK = topKs != null ? topKs[r] : 0;
      float topP = topPs != null ? topPs[r] : 1f;
      out[r] = this.sampleRow(logits, r, vocab, temperature, topK, topP, random);
    }
    return out;
  }

  /**
   * Draws with top-k disabled and top-p {@code 1} (temperature only).
   *
   * @param logits       {@code [rows, vocab]}
   * @param temperatures per-row softmax temperature
   * @param random       Gumbel draw source; must not be {@code null}
   * @return token ids, length {@code rows}
   */
  public int[] forward(final Tensor logits, final float[] temperatures, final Random random) {
    int[] topKs = new int[temperatures.length];
    float[] topPs = new float[temperatures.length];
    Arrays.fill(topPs, 1f);
    return this.forward(logits, temperatures, topKs, topPs, random);
  }

  /**
   * Softmax → top-k → top-p → Gumbel-max for one vocabulary row.
   */
  private int sampleRow(
    final Tensor logits,
    final int row,
    final int vocab,
    final float temperature,
    final int topK,
    final float topP,
    final Random random
  ) {
    if (topK == 1) {
      return this.argmax(logits, row, vocab);
    }

    float[] scores = this.softmaxRow(logits, row, vocab, temperature);
    if (topK > 0 && topK < vocab) {
      this.keepHighest(scores, topK);
      this.renormalize(scores);
    }
    if (topP < 1f) {
      this.keepNucleus(scores, topP);
      this.renormalize(scores);
    }
    return this.drawGumbel(scores, random);
  }

  private int argmax(final Tensor logits, final int row, final int vocab) {
    int base = logits.offset() + row * vocab;
    int best = 0;
    float bestScore = logits.data()[base];
    for (int i = 1; i < vocab; i++) {
      float v = logits.data()[base + i];
      if (v > bestScore) {
        bestScore = v;
        best = i;
      }
    }
    return best;
  }

  /**
   * Rescales {@code scores} to sum to 1. No-op when the mass is {@code <= 0}.
   */
  private void renormalize(final float[] scores) {
    float sum = 0f;
    for (float s : scores) {
      sum += s;
    }
    if (sum <= 0f) {
      return;
    }
    float inv = 1f / sum;
    for (int i = 0; i < scores.length; i++) {
      scores[i] *= inv;
    }
  }

  private float[] softmaxRow(
    final Tensor logits, final int row, final int vocab, final float temperature
  ) {
    float[] scores = new float[vocab];
    int base = logits.offset() + row * vocab;
    float[] data = logits.data();
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < vocab; i++) {
      float value = data[base + i] / temperature;
      scores[i] = value;
      if (value > max) {
        max = value;
      }
    }
    float sum = 0f;
    for (int i = 0; i < vocab; i++) {
      float exponent = (float) Math.exp(scores[i] - max);
      scores[i] = exponent;
      sum += exponent;
    }
    float inverse = 1f / sum;
    for (int i = 0; i < vocab; i++) {
      scores[i] *= inverse;
    }
    return scores;
  }

  private void keepHighest(final float[] scores, final int topK) {
    int[] best = this.topIndices(scores, topK);
    float[] kept = new float[topK];
    for (int i = 0; i < topK; i++) {
      kept[i] = scores[best[i]];
    }
    Arrays.fill(scores, 0f);
    for (int i = 0; i < topK; i++) {
      scores[best[i]] = kept[i];
    }
  }

  private void keepNucleus(final float[] scores, final float topP) {
    int[] heap = new int[scores.length];
    int size = 0;
    for (int index = 0; index < scores.length; index++) {
      if (scores[index] > 0f) {
        heap[size++] = index;
      }
    }
    if (size == 0) {
      return;
    }
    this.heapifyBestFirst(scores, heap, size);
    float cumulative = 0f;
    while (size > 0) {
      int index = this.pollBest(scores, heap, size);
      size--;
      cumulative += scores[index];
      if (cumulative >= topP) {
        this.zeroHeap(scores, heap, size);
        return;
      }
    }
  }

  private void heapifyBestFirst(final float[] scores, final int[] heap, final int size) {
    for (int index = size / 2 - 1; index >= 0; index--) {
      this.siftBestDown(scores, heap, size, index);
    }
  }

  private int pollBest(final float[] scores, final int[] heap, final int size) {
    int best = heap[0];
    int last = size - 1;
    heap[0] = heap[last];
    if (last > 0) {
      this.siftBestDown(scores, heap, last, 0);
    }
    return best;
  }

  private void siftBestDown(
    final float[] scores, final int[] heap, final int size, final int index
  ) {
    int node = index;
    while (true) {
      int left = node * 2 + 1;
      if (left >= size) {
        return;
      }
      int right = left + 1;
      int bestChild = left;
      if (right < size && this.better(scores, heap[right], heap[left])) {
        bestChild = right;
      }
      if (!this.better(scores, heap[bestChild], heap[node])) {
        return;
      }
      this.swap(heap, node, bestChild);
      node = bestChild;
    }
  }

  private void zeroHeap(final float[] scores, final int[] heap, final int size) {
    for (int i = 0; i < size; i++) {
      scores[heap[i]] = 0f;
    }
  }

  private int drawGumbel(final float[] scores, final Random random) {
    int best = 0;
    float bestScore = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < scores.length; i++) {
      float probability = scores[i];
      if (probability <= 0f) {
        continue;
      }
      float gumbel = (float) -Math.log(Math.max(1e-10, random.nextFloat()));
      float score = probability / Math.max(1e-10f, gumbel);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  private int[] topIndices(final float[] scores, final int k) {
    int[] heap = new int[k];
    int size = 0;
    for (int index = 0; index < scores.length; index++) {
      if (size < k) {
        heap[size] = index;
        this.siftUp(scores, heap, size);
        size++;
        continue;
      }
      if (this.better(scores, index, heap[0])) {
        heap[0] = index;
        this.siftDown(scores, heap, size, 0);
      }
    }
    return heap;
  }

  private void siftUp(final float[] scores, final int[] heap, final int index) {
    int node = index;
    while (node > 0) {
      int parent = (node - 1) / 2;
      if (!this.worse(scores, heap[node], heap[parent])) {
        break;
      }
      this.swap(heap, node, parent);
      node = parent;
    }
  }

  private void siftDown(final float[] scores, final int[] heap, final int size, final int index) {
    int node = index;
    while (true) {
      int left = node * 2 + 1;
      if (left >= size) {
        return;
      }
      int right = left + 1;
      int worst = left;
      if (right < size && this.worse(scores, heap[right], heap[left])) {
        worst = right;
      }
      if (!this.worse(scores, heap[worst], heap[node])) {
        return;
      }
      this.swap(heap, node, worst);
      node = worst;
    }
  }

  private boolean better(final float[] scores, final int left, final int right) {
    return this.worse(scores, right, left);
  }

  private boolean worse(final float[] scores, final int left, final int right) {
    float leftScore = scores[left];
    float rightScore = scores[right];
    if (leftScore != rightScore) {
      return leftScore < rightScore;
    }
    return left > right;
  }

  private void swap(final int[] values, final int left, final int right) {
    int held = values[left];
    values[left] = values[right];
    values[right] = held;
  }
}
