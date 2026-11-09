package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * One object from a YOLO26 detection, instance, pose, or oriented-box head.
 *
 * <p>{@code classId} is the winning class channel. {@link LlmModel#classNames()} is
 * {@code classes.txt} in that same order when a caller wants the name.
 *
 * @param classId     class channel from the head, {@code 0}-based
 * @param probability score in {@code [0, 1]}
 * @param shape       box, mask outline, or rotated corners
 * @param keypoints   visible pose joints in the model's order; empty for every other task
 * @since 1.5.0
 */
public record VisionHit(
  int classId,
  float probability,
  Shape2D shape,
  List<Keypoint> keypoints
) {

  /**
   * Hit with no pose points.
   *
   * @param classId     class channel; must be {@code >= 0}
   * @param probability score in {@code [0, 1]}
   * @param shape       geometry; must not be {@code null}
   */
  public VisionHit(final int classId, final float probability, final Shape2D shape) {
    this(classId, probability, shape, List.of());
  }

  /**
   * @throws NullPointerException     if a required value is {@code null}
   * @throws IllegalArgumentException if {@code classId} is negative or the score is outside
   *                                  {@code [0, 1]}
   */
  public VisionHit {
    requireNonNull(shape, "shape");
    requireNonNull(keypoints, "keypoints");
    if (classId < 0) {
      throw new IllegalArgumentException("classId must be >= 0");
    }
    if (!Float.isFinite(probability) || probability < 0f || probability > 1f) {
      throw new IllegalArgumentException("probability must be in [0, 1]");
    }
    keypoints = List.copyOf(keypoints);
  }
}
