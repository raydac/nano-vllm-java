package com.igormaznitsa.nanollvm.models.internal.yolo;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;
import java.util.Map;

/**
 * Named fused weights. A null map builds zeros of the requested shape so block tests can run
 * without an ONNX file.
 */
final class YoloTensors {

  static final String EXPORT =
    "yolo export model=yolo26n.pt format=onnx nms=False imgsz=640";

  private final Map<String, Tensor> weights;
  private final int syntheticStem;
  private final boolean syntheticDeep;

  private YoloTensors(
    final Map<String, Tensor> weights,
    final int syntheticStem,
    final boolean syntheticDeep
  ) {
    this.weights = weights;
    this.syntheticStem = syntheticStem;
    this.syntheticDeep = syntheticDeep;
  }

  static YoloTensors of(final Map<String, Tensor> weights) {
    return new YoloTensors(requireNonNull(weights, "weights"), 0, false);
  }

  static YoloTensors synthetic() {
    return synthetic(16, false);
  }

  static YoloTensors synthetic(final int stemChannels, final boolean deeper) {
    return new YoloTensors(null, stemChannels, deeper);
  }

  Tensor take(final String name, final int... shape) {
    if (this.weights == null) {
      return Tensor.zeros(shape);
    }
    Tensor found = this.find(name);
    if (found == null) {
      throw new ModelLoadException(
        "YOLO26 ONNX is missing fused weight " + name
          + ". Export the end-to-end head with: " + EXPORT);
    }
    if (!Arrays.equals(found.shape(), shape)) {
      throw new ModelLoadException(
        "YOLO26 weight " + name + " has shape " + Arrays.toString(found.shape())
          + ", expected " + Arrays.toString(shape));
    }
    return found;
  }

  Tensor optional(final String name, final int... shape) {
    if (this.weights == null) {
      return Tensor.zeros(shape);
    }
    Tensor found = this.find(name);
    if (found == null) {
      return null;
    }
    if (!Arrays.equals(found.shape(), shape)) {
      throw new ModelLoadException(
        "YOLO26 weight " + name + " has shape " + Arrays.toString(found.shape())
          + ", expected " + Arrays.toString(shape));
    }
    return found;
  }

  boolean has(final String name) {
    return this.weights != null && this.find(name) != null;
  }

  int[] shape(final String name) {
    Tensor found = this.weights == null ? null : this.find(name);
    if (found == null) {
      throw new ModelLoadException("YOLO26 ONNX is missing " + name + ". " + EXPORT);
    }
    return found.shape();
  }

  boolean hasDeeperBlock() {
    if (this.weights == null) {
      return this.syntheticDeep;
    }
    return this.weights.keySet().stream().anyMatch(name -> name.contains("model.2.m.1."));
  }

  int stemChannels() {
    if (this.weights == null) {
      return this.syntheticStem;
    }
    Tensor stem = this.find("model.0.conv.weight");
    if (stem == null || stem.shape().length != 4) {
      throw new ModelLoadException(
        "YOLO26 ONNX initializers are not fused module weights (expected model.0.conv.weight). "
          + EXPORT);
    }
    return stem.size(0);
  }

  private Tensor find(final String name) {
    Tensor exact = this.weights.get(name);
    if (exact != null) {
      return exact;
    }
    Tensor match = null;
    int matchLength = Integer.MAX_VALUE;
    for (Map.Entry<String, Tensor> entry : this.weights.entrySet()) {
      String key = entry.getKey();
      if (key.equals(name) || key.endsWith("." + name)) {
        if (key.length() < matchLength) {
          match = entry.getValue();
          matchLength = key.length();
        }
      }
    }
    return match;
  }
}
