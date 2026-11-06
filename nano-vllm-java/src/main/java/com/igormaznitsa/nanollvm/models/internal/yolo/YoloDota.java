package com.igormaznitsa.nanollvm.models.internal.yolo;

import java.util.List;

/**
 * DOTA names for a 15-class YOLO26 oriented-box head when no {@code classes.txt} is beside the file.
 */
final class YoloDota {

  static final List<String> NAMES = List.of(
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool");

  private YoloDota() {
  }
}
