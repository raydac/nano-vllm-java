package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Image payload for a YOLO26 forward via {@link LlmModel#generate(LlmInput, LlmModality)}.
 *
 * <p>{@link ArgbImage} copies the pixels, so the caller's array can change after construction.
 *
 * @param image packed {@code 0xAARRGGBB} raster; must be non-empty
 * @since 1.5.0
 */
public record LlmInImage(ArgbImage image) implements LlmInput {

  /**
   * @throws NullPointerException if {@code image} is {@code null}
   */
  public LlmInImage {
    requireNonNull(image, "image");
  }
}
