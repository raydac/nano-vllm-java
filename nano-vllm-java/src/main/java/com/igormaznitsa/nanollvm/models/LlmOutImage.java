package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Raster from a YOLO26 semantic or depth head.
 *
 * <p>Semantic pixels are class indexes stored as gray {@code 0xAARRGGBB} values. Depth pixels are
 * grayscale; {@link #minimum()} and {@link #maximum()} are the meters those shades stand for.
 *
 * @param image   result raster
 * @param minimum low end of the value range (class {@code 0}, or the nearest depth)
 * @param maximum high end of the value range (last class, or the farthest depth)
 * @since 1.5.0
 */
public record LlmOutImage(ArgbImage image, float minimum, float maximum)
  implements LlmOutVision {

  /**
   * @throws NullPointerException     if {@code image} is {@code null}
   * @throws IllegalArgumentException if the range is non-finite or inverted
   */
  public LlmOutImage {
    requireNonNull(image, "image");
    if (!Float.isFinite(minimum) || !Float.isFinite(maximum) || maximum < minimum) {
      throw new IllegalArgumentException("image range must be finite and ordered");
    }
  }

  @Override
  public LlmModality modality() {
    return LlmModality.IMAGE;
  }
}
