package com.igormaznitsa.nanollvm.models.internal.yolo;

import static com.igormaznitsa.nanollvm.layers.Conv2d.reusingPatches;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.LlmInImage;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmOutput;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import java.util.List;

/**
 * YOLO26 graph for nano through extra-large. Weights come from a fused end-to-end ONNX export.
 */
public final class Yolo26Model {

  private final YoloTask task;
  private final YoloScale scale;
  private final YoloTrunk trunk;
  private final YoloHead head;
  private final List<String> names;
  private final int classCount;

  private Yolo26Model(
    final YoloTask task,
    final YoloScale scale,
    final YoloTrunk trunk,
    final YoloHead head,
    final List<String> names,
    final int classCount
  ) {
    this.task = task;
    this.scale = scale;
    this.trunk = trunk;
    this.head = head;
    this.names = names;
    this.classCount = classCount;
  }

  static Yolo26Model load(
    final YoloTensors weights,
    final YoloTask task,
    final List<String> names
  ) {
    requireNonNull(task, "task");
    requireNonNull(names, "names");
    YoloScale scale = YoloScale.from(weights);
    YoloWidths width = YoloWidths.of(scale);
    return new Yolo26Model(
      task,
      scale,
      YoloTrunk.load(weights, width, task),
      YoloHead.load(weights, width, task),
      List.copyOf(names),
      YoloHead.declaredClasses(weights, task));
  }

  public static Yolo26Model synthetic(final YoloTask task) {
    List<String> names = java.util.stream.IntStream.range(0, Math.max(task.classes(), 1))
      .mapToObj(index -> "class-" + index)
      .toList();
    return load(YoloTensors.synthetic(), task, names);
  }

  public String architectureName() {
    return "yolo26" + this.scale.suffix + "-" + this.task.wireName();
  }

  public YoloTask task() {
    return this.task;
  }

  /**
   * Class axis of the loaded head. Depth is {@code 0}. A real file reads this from the class
   * tensor; the synthetic graph uses the published nano width.
   */
  public int classCount() {
    return this.classCount;
  }

  public List<String> classNames() {
    return List.copyOf(this.names);
  }

  public LlmModality outputModality() {
    return this.task == YoloTask.CLASSIFY ? LlmModality.LABELS : LlmModality.IMAGE;
  }

  public LlmOutput forward(final ArgbImage image, final MatmulRuntime runtime) {
    return this.forward(new LlmInImage(image), runtime);
  }

  public LlmOutput forward(
    final ArgbImage image,
    final float minProbability,
    final MatmulRuntime runtime
  ) {
    return this.forward(new LlmInImage(image, minProbability), runtime);
  }

  public LlmOutput forward(final LlmInImage input, final MatmulRuntime runtime) {
    requireNonNull(input, "input");
    return this.forwardLetter(
      YoloLetterbox.of(input.image(), input.imageSize()), input, runtime);
  }

  LlmOutput forwardLetter(final YoloLetterbox letter, final MatmulRuntime runtime) {
    return this.forwardLetter(letter, LlmInImage.DEFAULT_MIN_PROBABILITY, runtime);
  }

  LlmOutput forwardLetter(
    final YoloLetterbox letter,
    final float minProbability,
    final MatmulRuntime runtime
  ) {
    if (!Float.isFinite(minProbability) || minProbability < 0f || minProbability > 1f) {
      throw new IllegalArgumentException("minProbability must be in [0, 1]");
    }
    return this.forwardLetter(
      letter, new LlmInImage(new ArgbImage(1, 1, new int[] {0}), minProbability), runtime);
  }

  LlmOutput forwardLetter(
    final YoloLetterbox letter,
    final LlmInImage request,
    final MatmulRuntime runtime
  ) {
    this.requireImageRequest(request);
    MatmulRuntime kernels = runtime == null ? MatmulRuntime.sequential() : runtime;
    return reusingPatches(() -> this.head.read(
      this.trunk.forward(letter.tensor, kernels),
      letter,
      this.names,
      request,
      kernels));
  }

  private void requireImageRequest(final LlmInImage request) {
    if (!this.emitsObjects()
      && Float.compare(request.minProbability(), LlmInImage.DEFAULT_MIN_PROBABILITY) != 0) {
      throw new IllegalArgumentException(
        "minProbability applies to detection, segmentation, pose, and oriented boxes");
    }
    if (!request.classMinProbabilities().isEmpty() && !this.emitsObjects()) {
      throw new IllegalArgumentException(
          "addClass applies to detection, segmentation, pose, and oriented boxes");
    }
    if (this.emitsObjects()) {
      this.requireKnownClasses(request);
    }
    if (!this.emitsObjects() && request.maxDetections() != LlmInImage.DEFAULT_MAX_DETECTIONS) {
      throw new IllegalArgumentException(
        "maxDetections applies to detection, segmentation, pose, and oriented boxes");
    }
    if (this.task != YoloTask.POSE
      && Float.compare(
      request.keypointMinProbability(), LlmInImage.DEFAULT_KEYPOINT_MIN_PROBABILITY) != 0) {
      throw new IllegalArgumentException("keypointMinProbability applies to pose");
    }
    if (this.task != YoloTask.CLASSIFY && request.maxLabels() != LlmInImage.DEFAULT_MAX_LABELS) {
      throw new IllegalArgumentException("maxLabels applies to classification");
    }
  }

  private void requireKnownClasses(final LlmInImage request) {
    for (String name : request.classMinProbabilities().keySet()) {
      int index = this.names.indexOf(name);
      if (index < 0 || index >= this.classCount) {
        throw new IllegalArgumentException("unknown class: " + name);
      }
    }
  }

  private boolean emitsObjects() {
    return switch (this.task) {
      case DETECT, SEGMENT, POSE, OBB -> true;
      case SEMANTIC, DEPTH, CLASSIFY -> false;
    };
  }
}
