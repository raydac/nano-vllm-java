package com.igormaznitsa.nanollvm.models;

/**
 * Axis-aligned rectangle in image pixels. {@code right} is exclusive of {@code left} only when
 * the box has area; a zero-area box is still a valid point detection.
 *
 * @param left   smaller horizontal edge
 * @param top    smaller vertical edge
 * @param right  larger horizontal edge
 * @param bottom larger vertical edge
 * @since 1.5.0
 */
public record Box(float left, float top, float right, float bottom) implements Shape2D {

  /**
   * @throws IllegalArgumentException if an edge is non-finite or the box is inverted
   */
  public Box {
    if (!Float.isFinite(left) || !Float.isFinite(top)
      || !Float.isFinite(right) || !Float.isFinite(bottom)) {
      throw new IllegalArgumentException("box edges must be finite");
    }
    if (right < left || bottom < top) {
      throw new IllegalArgumentException("box is inverted");
    }
  }
}
