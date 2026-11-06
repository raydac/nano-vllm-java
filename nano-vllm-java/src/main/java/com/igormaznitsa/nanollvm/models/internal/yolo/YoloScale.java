package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;

/**
 * Nano and small compound scales. Wider scales use a different inner block and are rejected.
 */
final class YoloScale {

  final String suffix;
  private final float depth;
  private final float width;
  private final int maxChannels;

  private YoloScale(final String suffix, final float depth, final float width,
                    final int maxChannels) {
    this.suffix = suffix;
    this.depth = depth;
    this.width = width;
    this.maxChannels = maxChannels;
  }

  static YoloScale ofStem(final int stemChannels) {
    if (stemChannels == 16) {
      return new YoloScale("n", 0.50f, 0.25f, 1024);
    }
    if (stemChannels == 32) {
      return new YoloScale("s", 0.50f, 0.50f, 1024);
    }
    throw new ModelLoadException(
      "YOLO26 scale is not nano or small (stem channels " + stemChannels
        + "). m, l, and x are not loaded. " + YoloTensors.EXPORT);
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
