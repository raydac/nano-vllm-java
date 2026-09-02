package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Huffman hierarchical softmax rebuilt from label counts at load (matches Meta fastText).
 */
final class HierarchicalSoftmax {

  private final int labelCount;
  private final Node[] tree;
  private final Matrix output;

  private HierarchicalSoftmax(final int labelCount, final Node[] tree, final Matrix output) {
    this.labelCount = labelCount;
    this.tree = tree;
    this.output = output;
  }

  static HierarchicalSoftmax build(final long[] labelCounts, final Matrix output) {
    final int osz = labelCounts.length;
    if (osz <= 0) {
      throw new IllegalArgumentException("hierarchical softmax needs at least one label");
    }
    final Node[] tree = new Node[2 * osz - 1];
    for (int i = 0; i < tree.length; i++) {
      tree[i] = new Node();
    }
    for (int i = 0; i < osz; i++) {
      tree[i].count = labelCounts[i];
    }
    int leaf = osz - 1;
    int node = osz;
    for (int i = osz; i < 2 * osz - 1; i++) {
      final int[] mini = new int[2];
      for (int j = 0; j < 2; j++) {
        if (leaf >= 0 && tree[leaf].count < tree[node].count) {
          mini[j] = leaf--;
        } else {
          mini[j] = node++;
        }
      }
      tree[i].left = mini[0];
      tree[i].right = mini[1];
      tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    }
    return new HierarchicalSoftmax(osz, tree, output);
  }

  private static double stdLog(final double value) {
    return Math.log(value + 1e-5);
  }

  List<FastTextModel.Prediction> predict(
    final float[] hidden,
    final Dictionary dictionary,
    final int k,
    final float threshold
  ) {
    final PriorityQueue<Candidate> heap = new PriorityQueue<>(
      Comparator.comparingDouble(Candidate::logScore));
    this.dfs(k, threshold, 2 * this.labelCount - 2, 0.0, heap, hidden);

    final List<FastTextModel.Prediction> result = new ArrayList<>(heap.size());
    while (!heap.isEmpty()) {
      final Candidate candidate = heap.poll();
      result.add(new FastTextModel.Prediction(
        dictionary.getLabel(candidate.labelId),
        (float) Math.exp(candidate.logScore)));
    }
    result.sort(Comparator.comparingDouble(FastTextModel.Prediction::probability).reversed());
    return List.copyOf(result);
  }

  private void dfs(
    final int k,
    final float threshold,
    final int node,
    final double score,
    final PriorityQueue<Candidate> heap,
    final float[] hidden
  ) {
    if (score < stdLog(threshold)) {
      return;
    }
    if (heap.size() == k && score < heap.peek().logScore) {
      return;
    }

    final Node current = this.tree[node];
    if (current.left == -1 && current.right == -1) {
      heap.offer(new Candidate(score, node));
      if (heap.size() > k) {
        heap.poll();
      }
      return;
    }

    float probability = this.sigmoid(this.output.dotRow(hidden, node - this.labelCount));
    this.dfs(k, threshold, current.left, score + stdLog(1.0 - probability), heap, hidden);
    this.dfs(k, threshold, current.right, score + stdLog(probability), heap, hidden);
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

  private static final class Node {
    int left = -1;
    int right = -1;
    long count = 1_000_000_000_000_000L;
  }

  private record Candidate(double logScore, int labelId) {
  }
}
