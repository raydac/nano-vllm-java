package com.igormaznitsa.nanollvm.models;

/**
 * Geometry returned with a detection. {@link Box} is an axis-aligned rectangle.
 * {@link Polygon} is an outline of one or more points.
 *
 * @since 1.5.0
 */
public sealed interface Shape2D permits Box, Polygon {
}
