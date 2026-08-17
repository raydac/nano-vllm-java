package com.igormaznitsa.nanollvm.models.llmarch;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxTransport;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Shared container-to-bag decode used by architecture processors.
 *
 * @since 1.1.0
 */
final class ArchitectureFills {

  private ArchitectureFills() {
  }

  /**
   * Reads GGUF payloads (packed or unpacked) and checks schema names.
   *
   * @since 1.1.0
   */
  static WeightBag gguf(
    final GgufTransport transport,
    final BoundModel bound,
    final LlmListener streams,
    final boolean allowUnpackGguf
  ) {
    WeightBag weights = new WeightBag(transport.readPayloads(allowUnpackGguf, streams));
    bound.requireLoadedWeights(weights);
    return weights;
  }

  /**
   * Reads ONNX float initializers, remaps names, and checks schema names.
   *
   * @since 1.1.0
   */
  static WeightBag onnx(
    final OnnxTransport transport,
    final BoundModel bound,
    final LlmListener streams,
    final boolean embedding
  ) throws IOException {
    Map<String, Tensor> containerTensors = transport.readFloatingTensors(streams);
    Map<String, Tensor> named = new LinkedHashMap<>();
    Map<String, String> aliases = transport.matMulAliases();
    for (var entry : containerTensors.entrySet()) {
      Tensor tensor = entry.getValue();
      String alias = aliases.get(entry.getKey());
      String sourceName = alias != null ? alias : entry.getKey();
      String mapped = embedding
        ? OnnxWeightNames.normalizeBertName(sourceName)
        : OnnxWeightNames.normalizeChatName(sourceName);
      named.put(mapped, alias != null ? transposeIfMatrix(tensor) : tensor);
    }
    return ModelLoader.assembleFromNamedTensors(
      named, transport.label(), bound.config(), bound.schema(), streams);
  }

  static Tensor transposeIfMatrix(final Tensor tensor) {
    if (tensor.shape().length != 2) {
      return tensor;
    }
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    float[] src = tensor.toFloatArray();
    float[] dst = new float[src.length];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        dst[c * rows + r] = src[r * cols + c];
      }
    }
    return Tensor.of(dst, cols, rows);
  }
}
