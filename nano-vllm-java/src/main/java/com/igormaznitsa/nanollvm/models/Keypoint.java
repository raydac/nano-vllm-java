package com.igormaznitsa.nanollvm.models;

/**
 * One visible body joint from a pose head. Joints follow the 17-point COCO order
 * (0 nose through 16 right ankle). A joint whose confidence is below one half is omitted,
 * so a limb can be drawn only when both of its joints are in the list.
 *
 * @param joint index in that order, {@code 0} through {@code 16}
 * @param x     horizontal pixel in the original picture
 * @param y     vertical pixel in the original picture
 * @since 1.5.0
 */
public record Keypoint(int joint, float x, float y) {

  public static final int COUNT = 17;

  /**
   * @throws IllegalArgumentException if the joint is outside the COCO body or a coordinate is non-finite
   */
  public Keypoint {
    if (joint < 0 || joint >= COUNT) {
      throw new IllegalArgumentException("joint must be in 0.." + (COUNT - 1));
    }
    if (!Float.isFinite(x) || !Float.isFinite(y)) {
      throw new IllegalArgumentException("keypoint coordinates must be finite");
    }
  }
}
