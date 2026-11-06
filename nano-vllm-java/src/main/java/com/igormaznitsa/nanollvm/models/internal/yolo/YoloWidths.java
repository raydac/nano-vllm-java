package com.igormaznitsa.nanollvm.models.internal.yolo;

/**
 * Channel widths for one YOLO26 compound scale.
 */
final class YoloWidths {

  final YoloScale scale;
  private final int c64;
  private final int c128;
  private final int c256;
  private final int c512;
  private final int c1024;

  private YoloWidths(
    final YoloScale scale,
    final int c64,
    final int c128,
    final int c256,
    final int c512,
    final int c1024
  ) {
    this.scale = scale;
    this.c64 = c64;
    this.c128 = c128;
    this.c256 = c256;
    this.c512 = c512;
    this.c1024 = c1024;
  }

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

  int c64() {
    return this.c64;
  }

  int c128() {
    return this.c128;
  }

  int c256() {
    return this.c256;
  }

  int c512() {
    return this.c512;
  }

  int c1024() {
    return this.c1024;
  }
}
