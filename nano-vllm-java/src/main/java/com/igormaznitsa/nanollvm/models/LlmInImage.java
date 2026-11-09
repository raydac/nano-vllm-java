package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Image payload for a YOLO26 forward via {@link LlmModel#generate(LlmInput, LlmModality)}.
 *
 * <p>{@link ArgbImage} copies the pixels, so the caller's array can change after construction.
 * Object heads (detection, instance segmentation, pose, oriented boxes) keep a hit only when its
 * score is at least {@link #minProbability()} and it is among the first {@link #maxDetections()}
 * after ranking. Pose drops a joint whose confidence is below {@link #keypointMinProbability()}.
 * Classification returns at most {@link #maxLabels()} names. {@link #imageSize()} is the square
 * letterbox the file is run at. The one- and two-argument constructors use the defaults. A knob
 * that does not apply to the loaded head is rejected when it is not the default, so a caller
 * cannot think a semantic map was filtered by an object score.
 *
 * @param image                   packed {@code 0xAARRGGBB} raster; must be non-empty
 * @param minProbability          lowest kept object score, in {@code [0, 1]}
 * @param imageSize               square letterbox edge, a multiple of 32 and at least 32
 * @param maxDetections           how many ranked objects an object head may return
 * @param keypointMinProbability  lowest kept pose-joint confidence, in {@code [0, 1]}
 * @param maxLabels               how many classification labels to return
 * @since 1.5.0
 */
public record LlmInImage(
  ArgbImage image,
  float minProbability,
  int imageSize,
  int maxDetections,
  float keypointMinProbability,
  int maxLabels
) implements LlmInput {

  /**
   * Lowest object score when the caller passes only the picture.
   */
  public static final float DEFAULT_MIN_PROBABILITY = 0.25f;

  /**
   * Square letterbox used when the caller does not set one. Matches a fused export at
   * {@code imgsz=640}.
   */
  public static final int DEFAULT_IMAGE_SIZE = 640;

  /**
   * Ranked objects kept when the caller does not set a cap.
   */
  public static final int DEFAULT_MAX_DETECTIONS = 300;

  /**
   * Pose-joint confidence kept when the caller does not set one.
   */
  public static final float DEFAULT_KEYPOINT_MIN_PROBABILITY = 0.5f;

  /**
   * Classification labels returned when the caller does not set a count.
   */
  public static final int DEFAULT_MAX_LABELS = 5;

  /**
   * Picture with every default knob.
   *
   * @param image packed raster; must not be {@code null}
   */
  public LlmInImage(final ArgbImage image) {
    this(image, DEFAULT_MIN_PROBABILITY);
  }

  /**
   * Picture with an object-score floor and the other default knobs.
   *
   * @param image          packed raster; must not be {@code null}
   * @param minProbability lowest kept object score, in {@code [0, 1]}
   */
  public LlmInImage(final ArgbImage image, final float minProbability) {
    this(
      image,
      minProbability,
      DEFAULT_IMAGE_SIZE,
      DEFAULT_MAX_DETECTIONS,
      DEFAULT_KEYPOINT_MIN_PROBABILITY,
      DEFAULT_MAX_LABELS);
  }

  /**
   * @throws NullPointerException     if {@code image} is {@code null}
   * @throws IllegalArgumentException if a score is outside {@code [0, 1]}, {@code imageSize} is
   *                                  not a multiple of 32, or a cap is below 1
   */
  public LlmInImage {
    requireNonNull(image, "image");
    if (!Float.isFinite(minProbability) || minProbability < 0f || minProbability > 1f) {
      throw new IllegalArgumentException("minProbability must be in [0, 1]");
    }
    if (imageSize < 32 || imageSize % 32 != 0) {
      throw new IllegalArgumentException("imageSize must be a multiple of 32 and at least 32");
    }
    if (maxDetections < 1) {
      throw new IllegalArgumentException("maxDetections must be >= 1");
    }
    if (!Float.isFinite(keypointMinProbability)
      || keypointMinProbability < 0f
      || keypointMinProbability > 1f) {
      throw new IllegalArgumentException("keypointMinProbability must be in [0, 1]");
    }
    if (maxLabels < 1) {
      throw new IllegalArgumentException("maxLabels must be >= 1");
    }
  }

  /**
   * Starts a picture whose letterbox, caps, and score floors are set by name.
   *
   * @param image packed raster; must not be {@code null}
   * @return a builder that still holds the defaults
   */
  public static Builder builder(final ArgbImage image) {
    return new Builder(image);
  }

  /**
   * Named knobs for {@link LlmInImage}. Unset fields stay at the defaults.
   */
  public static final class Builder {

    private final ArgbImage image;
    private float minProbability = DEFAULT_MIN_PROBABILITY;
    private int imageSize = DEFAULT_IMAGE_SIZE;
    private int maxDetections = DEFAULT_MAX_DETECTIONS;
    private float keypointMinProbability = DEFAULT_KEYPOINT_MIN_PROBABILITY;
    private int maxLabels = DEFAULT_MAX_LABELS;

    private Builder(final ArgbImage image) {
      this.image = requireNonNull(image, "image");
    }

    /**
     * @param minProbability lowest kept object score, in {@code [0, 1]}
     * @return {@code this}
     */
    public Builder minProbability(final float minProbability) {
      this.minProbability = minProbability;
      return this;
    }

    /**
     * @param imageSize square letterbox edge, a multiple of 32 and at least 32
     * @return {@code this}
     */
    public Builder imageSize(final int imageSize) {
      this.imageSize = imageSize;
      return this;
    }

    /**
     * @param maxDetections how many ranked objects an object head may return; at least 1
     * @return {@code this}
     */
    public Builder maxDetections(final int maxDetections) {
      this.maxDetections = maxDetections;
      return this;
    }

    /**
     * @param keypointMinProbability lowest kept pose-joint confidence, in {@code [0, 1]}
     * @return {@code this}
     */
    public Builder keypointMinProbability(final float keypointMinProbability) {
      this.keypointMinProbability = keypointMinProbability;
      return this;
    }

    /**
     * @param maxLabels how many classification labels to return; at least 1
     * @return {@code this}
     */
    public Builder maxLabels(final int maxLabels) {
      this.maxLabels = maxLabels;
      return this;
    }

    /**
     * @return the image input
     * @throws IllegalArgumentException if a knob is out of range
     */
    public LlmInImage build() {
      return new LlmInImage(
        this.image,
        this.minProbability,
        this.imageSize,
        this.maxDetections,
        this.keypointMinProbability,
        this.maxLabels);
    }
  }
}
