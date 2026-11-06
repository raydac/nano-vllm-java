package com.igormaznitsa.nanollvm.models;

/**
 * A point in image pixels. This is not {@code java.awt.geom.Point2D}.
 *
 * @param x horizontal pixel
 * @param y vertical pixel
 * @since 1.5.0
 */
public record Point2D(float x, float y) {

  /**
   * @throws IllegalArgumentException if either coordinate is non-finite
   */
  public Point2D {
    if (!Float.isFinite(x) || !Float.isFinite(y)) {
      throw new IllegalArgumentException("point coordinates must be finite");
    }
  }
}
