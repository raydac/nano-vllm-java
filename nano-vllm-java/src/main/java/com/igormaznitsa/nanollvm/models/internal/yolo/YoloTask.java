package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
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
    if (name.endsWith(".pt") || name.contains("-p2") || name.contains("-p6")
      || name.contains("yolo26m") || name.contains("yolo26l") || name.contains("yolo26x")) {
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
    if (name.contains("yolo26n") || name.contains("yolo26s")) {
      return DETECT;
    }
    return null;
  }

  private static UnsupportedModelException rejected(final String fileName) {
    return new UnsupportedModelException(
      "YOLO26 file is not a nano or small ONNX export: " + fileName
        +
        ". Use yolo26n or yolo26s with nms=False. P2, P6, YOLOE, and the .pt pickle are not loaded."
        + System.lineSeparator() + System.lineSeparator() + ModelSupport.CATALOG,
      "yolo26",
      java.util.List.of(fileName));
  }

  public String wireName() {
    return this.wireName;
  }

  public int classes() {
    return switch (this) {
      case SEMANTIC -> 19;
      case CLASSIFY -> 1000;
      case DEPTH -> 0;
      default -> 80;
    };
  }
}
