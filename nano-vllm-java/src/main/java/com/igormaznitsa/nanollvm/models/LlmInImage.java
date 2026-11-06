package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Image payload for a YOLO26 forward via {@link LlmModel#generate(LlmInput, LlmModality)}.
 *
 * <p>{@link ArgbImage} copies the pixels, so the caller's array can change after construction.
 * {@link #minProbability()} is the lowest object score kept by detection, instance segmentation,
 * pose, and oriented boxes. Semantic maps, depth, and classification have no object score; a value
 * other than {@link #DEFAULT_MIN_PROBABILITY} is rejected for those files.
 *
 * @param image          packed {@code 0xAARRGGBB} raster; must be non-empty
 * @param minProbability lowest kept object score, in {@code [0, 1]}
 * @since 1.5.0
 */
public record LlmInImage(ArgbImage image, float minProbability) implements LlmInput {

  /**
   * Lowest object score when the caller passes only the picture.
   */
  public static final float DEFAULT_MIN_PROBABILITY = 0.25f;

  /**
   * Picture with {@link #DEFAULT_MIN_PROBABILITY}.
   *
   * @param image packed raster; must not be {@code null}
   */
  public LlmInImage(final ArgbImage image) {
    this(image, DEFAULT_MIN_PROBABILITY);
  }

  /**
   * @throws NullPointerException     if {@code image} is {@code null}
   * @throws IllegalArgumentException if {@code minProbability} is non-finite or outside
   *                                  {@code [0, 1]}
   */
  public LlmInImage {
    requireNonNull(image, "image");
    if (!Float.isFinite(minProbability) || minProbability < 0f || minProbability > 1f) {
      throw new IllegalArgumentException("minProbability must be in [0, 1]");
    }
  }
}
