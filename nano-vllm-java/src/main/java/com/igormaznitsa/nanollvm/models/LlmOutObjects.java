package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Objects from a YOLO26 detection, instance, pose, or oriented-box head.
 *
 * @param objects hits that passed the confidence filter; may be empty
 * @since 1.5.0
 */
public record LlmOutObjects(List<VisionHit> objects) implements LlmOutVision {

  /**
   * @throws NullPointerException if {@code objects} or an element is {@code null}
   */
  public LlmOutObjects {
    requireNonNull(objects, "objects");
    objects = List.copyOf(objects);
  }

  @Override
  public LlmModality modality() {
    return LlmModality.IMAGE;
  }
}
