package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;

/**
 * Compound scales from the published YOLO26 YAML. Medium and larger force a {@code C3k} inner
 * block. Large and extra-large also repeat those blocks.
 */
final class YoloScale {

  final String suffix;
  private final float depth;
  private final float width;
  private final int maxChannels;
  private final boolean c3k;

  private YoloScale(
    final String suffix,
    final float depth,
    final float width,
    final int maxChannels,
    final boolean c3k
  ) {
    this.suffix = suffix;
    this.depth = depth;
    this.width = width;
    this.maxChannels = maxChannels;
    this.c3k = c3k;
  }

  static YoloScale from(final YoloTensors weights) {
    int stem = weights.stemChannels();
    if (stem == 16) {
      return scale("n", 0.50f, 0.25f, 1024, false);
    }
    if (stem == 32) {
      return scale("s", 0.50f, 0.50f, 1024, false);
    }
    if (stem == 64) {
      return weights.hasDeeperBlock()
        ? scale("l", 1.00f, 1.00f, 512, true)
        : scale("m", 0.50f, 1.00f, 512, true);
    }
    if (stem == 96) {
      return scale("x", 1.00f, 1.50f, 512, true);
    }
    throw new ModelLoadException(
      "YOLO26 stem has " + stem
        + " channels. Expected 16 (n), 32 (s), 64 (m or l), or 96 (x). " + YoloTensors.EXPORT);
  }

  private static YoloScale scale(
    final String suffix,
    final float depth,
    final float width,
    final int maxChannels,
    final boolean c3k
  ) {
    return new YoloScale(suffix, depth, width, maxChannels, c3k);
  }

  boolean usesC3k() {
    return this.c3k;
  }

  float depth() {
    return this.depth;
  }

  float width() {
    return this.width;
  }

  int maxChannels() {
    return this.maxChannels;
  }
}
