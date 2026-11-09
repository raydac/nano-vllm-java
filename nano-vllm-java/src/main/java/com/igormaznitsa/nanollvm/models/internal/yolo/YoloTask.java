package com.igormaznitsa.nanollvm.models.internal.yolo;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.models.LlmModelType;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import java.util.Locale;

/**
 * Which YOLO26 head a file name refers to.
 */
public enum YoloTask {
  DETECT("detect"),
  SEGMENT("seg"),
  SEMANTIC("sem"),
  DEPTH("depth"),
  CLASSIFY("cls"),
  POSE("pose"),
  OBB("obb");

  private final String wireName;

  YoloTask(final String wireName) {
    this.wireName = wireName;
  }

  public static YoloTask fromType(final LlmModelType type) {
    return switch (requireNonNull(type, "type")) {
      case YOLO_DETECT -> DETECT;
      case YOLO_SEGMENT -> SEGMENT;
      case YOLO_SEMANTIC -> SEMANTIC;
      case YOLO_DEPTH -> DEPTH;
      case YOLO_CLASSIFY -> CLASSIFY;
      case YOLO_POSE -> POSE;
      case YOLO_OBB -> OBB;
      case GGUF, HUGGING_FACE, PIPER, FASTTEXT ->
        throw new IllegalArgumentException("not a YOLO head: " + type);
    };
  }

  /**
   * @return the task, or {@code null} when the name is not a YOLO26 checkpoint
   */
  public static YoloTask fromFileName(final String fileName) {
    if (fileName == null || fileName.isBlank()) {
      return null;
    }
    String name = fileName.toLowerCase(Locale.ROOT);
    if (name.contains("yoloe")) {
      throw rejected(fileName);
    }
    if (!name.contains("yolo26")) {
      return null;
    }
    if (name.endsWith(".pt") || name.contains("-p2") || name.contains("-p6")) {
      throw rejected(fileName);
    }
    if (name.contains("-sem")) {
      return SEMANTIC;
    }
    if (name.contains("-seg")) {
      return SEGMENT;
    }
    if (name.contains("-depth")) {
      return DEPTH;
    }
    if (name.contains("-cls")) {
      return CLASSIFY;
    }
    if (name.contains("-pose")) {
      return POSE;
    }
    if (name.contains("-obb")) {
      return OBB;
    }
    if (name.contains("yolo26n") || name.contains("yolo26s") || name.contains("yolo26m")
      || name.contains("yolo26l") || name.contains("yolo26x")) {
      return DETECT;
    }
    return null;
  }

  private static UnsupportedModelException rejected(final String fileName) {
    return new UnsupportedModelException(
      "YOLO26 file is not a fused ONNX export: " + fileName
        + ". Use yolo26n, yolo26s, yolo26m, yolo26l, or yolo26x with nms=False. "
        + "P2, P6, YOLOE, and the .pt pickle are not loaded."
        + System.lineSeparator() + System.lineSeparator() + ModelSupport.CATALOG,
      "yolo26",
      java.util.List.of(fileName));
  }

  public String wireName() {
    return this.wireName;
  }

  /**
   * Width of the synthetic test graph. A real file's class count is the output axis of its class
   * tensor, not this value.
   */
  int classes() {
    return switch (this) {
      case SEMANTIC -> 19;
      case CLASSIFY -> 1000;
      case DEPTH -> 0;
      default -> 80;
    };
  }
}
