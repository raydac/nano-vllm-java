package com.igormaznitsa.nanollvm.models;

/**
 * Image-model result. Detection, instance segmentation, pose, and oriented boxes are
 * {@link LlmOutObjects}. Semantic segmentation and depth are {@link LlmOutImage}.
 *
 * @since 1.5.0
 */
public sealed interface LlmOutVision extends LlmOutput permits LlmOutObjects, LlmOutImage {
}
