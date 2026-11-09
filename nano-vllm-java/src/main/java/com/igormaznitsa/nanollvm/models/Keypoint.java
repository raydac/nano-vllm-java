package com.igormaznitsa.nanollvm.models;

/**
 * One visible body joint from a pose head. {@link #joint()} is the model's own index.
 * {@link #COUNT} is the 17-point COCO body (0 nose through 16 right ankle) a caller may use when
 * drawing that skeleton. A joint the caller filtered out is absent, so a limb is drawn only when
 * both of its joints are in the list. The confidence floor is {@link LlmInImage#keypointMinProbability()}.
 *
 * @param joint index in the model's joint order, {@code >= 0}
 * @param x     horizontal pixel in the original picture
 * @param y     vertical pixel in the original picture
 * @since 1.5.0
 */
public record Keypoint(int joint, float x, float y) {

  /**
   * Joints in the COCO body. A pose file may emit a different count; this constant does not
   * limit {@link #joint()}.
   */
  public static final int COUNT = 17;

  /**
   * @throws IllegalArgumentException if the joint is negative or a coordinate is non-finite
   */
  public Keypoint {
    if (joint < 0) {
      throw new IllegalArgumentException("joint must be >= 0");
    }
    if (!Float.isFinite(x) || !Float.isFinite(y)) {
      throw new IllegalArgumentException("keypoint coordinates must be finite");
    }
  }
}
