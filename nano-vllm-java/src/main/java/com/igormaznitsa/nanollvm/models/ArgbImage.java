package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Row-major pixels, {@code 0xAARRGGBB}, one {@code int} per pixel.
 * Index is {@code y * width + x}.
 *
 * <p>This is the picture type for {@link LlmInImage} and {@link LlmOutImage}. It does not use
 * {@code java.desktop}. A {@code BufferedImage} converts with {@code getRGB} into this layout.
 *
 * @param width  pixel columns, at least 1
 * @param height pixel rows, at least 1
 * @param pixels packed pixels, length {@code width * height}
 * @since 1.5.0
 */
@SuppressWarnings("ArrayRecordComponent")
public record ArgbImage(int width, int height, int[] pixels) {

  /**
   * @throws NullPointerException     if {@code pixels} is {@code null}
   * @throws IllegalArgumentException if the raster is empty or the buffer length is not
   *                                  {@code width * height}
   */
  public ArgbImage {
    if (width < 1 || height < 1) {
      throw new IllegalArgumentException("image must be non-empty");
    }
    requireNonNull(pixels, "pixels");
    if (pixels.length != (long) width * height) {
      throw new IllegalArgumentException("pixels length must be width * height");
    }
    pixels = pixels.clone();
  }

  @Override
  public int[] pixels() {
    return this.pixels.clone();
  }
}
