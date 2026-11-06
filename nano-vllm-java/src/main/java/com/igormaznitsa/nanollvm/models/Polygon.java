package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Closed outline in image pixels. The first point is not repeated at the end.
 *
 * @param points vertices in order; at least three
 * @since 1.5.0
 */
public record Polygon(List<Point2D> points) implements Shape2D {

  /**
   * @throws NullPointerException     if {@code points} or an element is {@code null}
   * @throws IllegalArgumentException if fewer than three vertices are present
   */
  public Polygon {
    requireNonNull(points, "points");
    if (points.size() < 3) {
      throw new IllegalArgumentException("polygon needs at least three points");
    }
    points = List.copyOf(points);
  }
}
