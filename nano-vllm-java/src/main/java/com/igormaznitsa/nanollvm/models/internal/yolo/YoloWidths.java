package com.igormaznitsa.nanollvm.models.internal.yolo;

/**
 * Channel widths for one YOLO26 compound scale.
 */
record YoloWidths(
  YoloScale scale,
  int c64,
  int c128,
  int c256,
  int c512,
  int c1024
) {

  static YoloWidths of(final YoloScale scale) {
    return new YoloWidths(
      scale,
      YoloMath.channels(64, scale.width(), scale.maxChannels()),
      YoloMath.channels(128, scale.width(), scale.maxChannels()),
      YoloMath.channels(256, scale.width(), scale.maxChannels()),
      YoloMath.channels(512, scale.width(), scale.maxChannels()),
      YoloMath.channels(1024, scale.width(), scale.maxChannels()));
  }

  int repeat(final int count) {
    return YoloMath.repeats(count, this.scale.depth());
  }
}
