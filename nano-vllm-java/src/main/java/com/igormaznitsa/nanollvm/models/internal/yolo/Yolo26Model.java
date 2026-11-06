package com.igormaznitsa.nanollvm.models.internal.yolo;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmOutput;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import java.util.List;

/**
 * YOLO26 nano or small graph. Weights come from a fused end-to-end ONNX export.
 */
public final class Yolo26Model {

  static final float CONFIDENCE = 0.25f;
  static final int MAX_DET = 300;

  private final YoloTask task;
  private final YoloScale scale;
  private final YoloTrunk trunk;
  private final YoloHead head;
  private final List<String> names;

  private Yolo26Model(
    final YoloTask task,
    final YoloScale scale,
    final YoloTrunk trunk,
    final YoloHead head,
    final List<String> names
  ) {
    this.task = task;
    this.scale = scale;
    this.trunk = trunk;
    this.head = head;
    this.names = names;
  }

  static Yolo26Model load(
    final YoloTensors weights,
    final YoloTask task,
    final List<String> names
  ) {
    requireNonNull(task, "task");
    requireNonNull(names, "names");
    YoloScale scale = YoloScale.ofStem(weights.stemChannels());
    YoloWidths width = YoloWidths.of(scale);
    return new Yolo26Model(
      task,
      scale,
      YoloTrunk.load(weights, width, task),
      YoloHead.load(weights, width, task),
      List.copyOf(names));
  }

  public static Yolo26Model synthetic(final YoloTask task) {
    List<String> names =
      task == YoloTask.CLASSIFY || task == YoloTask.SEMANTIC || task == YoloTask.DEPTH
        ? java.util.stream.IntStream.range(0, Math.max(task.classes(), 1))
        .mapToObj(index -> "class-" + index)
        .toList()
        : YoloCoco.NAMES;
    return load(YoloTensors.synthetic(), task, names);
  }

  public String architectureName() {
    return "yolo26" + this.scale.suffix + "-" + this.task.wireName();
  }

  public YoloTask task() {
    return this.task;
  }

  public LlmModality outputModality() {
    return this.task == YoloTask.CLASSIFY ? LlmModality.LABELS : LlmModality.IMAGE;
  }

  public LlmOutput forward(final ArgbImage image, final MatmulRuntime runtime) {
    requireNonNull(image, "image");
    return this.forwardLetter(YoloLetterbox.of(image), runtime);
  }

  LlmOutput forwardLetter(final YoloLetterbox letter, final MatmulRuntime runtime) {
    MatmulRuntime kernels = runtime == null ? MatmulRuntime.sequential() : runtime;
    return this.head.read(this.trunk.forward(letter.tensor, kernels), letter, this.names, kernels);
  }
}
