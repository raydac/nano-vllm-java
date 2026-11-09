package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * One object from a YOLO26 detection, instance, pose, or oriented-box head.
 *
 * @param className   label from the checkpoint's class list
 * @param probability score in {@code [0, 1]}
 * @param shape       box, mask outline, or rotated corners
 * @param keypoints   visible pose joints in COCO order; empty for every other task
 * @since 1.5.0
 */
public record VisionHit(
  String className,
  float probability,
  Shape2D shape,
  List<Keypoint> keypoints
) {

  /**
   * Hit with no pose points.
   *
   * @param className   label; must not be blank
   * @param probability score in {@code [0, 1]}
   * @param shape       geometry; must not be {@code null}
   */
  public VisionHit(final String className, final float probability, final Shape2D shape) {
    this(className, probability, shape, List.of());
  }

  /**
   * @throws NullPointerException     if a required value is {@code null}
   * @throws IllegalArgumentException if the name is blank or the score is outside {@code [0, 1]}
   */
  public VisionHit {
    requireNonNull(className, "className");
    requireNonNull(shape, "shape");
    requireNonNull(keypoints, "keypoints");
    if (className.isBlank()) {
      throw new IllegalArgumentException("className must not be blank");
    }
    if (!Float.isFinite(probability) || probability < 0f || probability > 1f) {
      throw new IllegalArgumentException("probability must be in [0, 1]");
    }
    keypoints = List.copyOf(keypoints);
  }
}
