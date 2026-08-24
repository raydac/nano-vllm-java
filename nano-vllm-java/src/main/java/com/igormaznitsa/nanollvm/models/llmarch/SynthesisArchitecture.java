package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.models.internal.TextToSpeech;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxTransport;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Template for text-to-speech families: ONNX initializer fill without chat/BERT remaps,
 * {@link #createSynthesis} required, never a chat, embedding, or speech graph.
 *
 * @since 1.3.0
 */
abstract sealed class SynthesisArchitecture implements ArchitectureProcessor
  permits PiperProcessor {

  @Override
  public final boolean isEmbedding() {
    return false;
  }

  @Override
  public final boolean isSynthesis() {
    return true;
  }

  @Override
  public BoundModel bind(final ContainerCatalog catalog, final ModelSupport.Selection selected) {
    return this.bindHf(selected, ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  @Override
  public WeightBag fill(
    final ContainerTransport transport,
    final BoundModel bound,
    final LlmListener io,
    final boolean allowUnpackGguf
  ) throws IOException {
    if (!(transport instanceof OnnxTransport onnx)) {
      throw new IllegalStateException(this.architectureId() + " loads from ONNX only");
    }
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Map<String, Tensor> tensors = new LinkedHashMap<>(onnx.readFloatingTensors(streams));
    Map<String, ConvLayout> layouts = new LinkedHashMap<>(onnx.convLayouts());
    this.copyIdentityAliases(onnx, tensors, layouts);
    this.remapWeights(tensors, layouts);
    return new WeightBag(tensors, layouts);
  }

  private void copyIdentityAliases(
    final OnnxTransport onnx,
    final Map<String, Tensor> tensors,
    final Map<String, ConvLayout> layouts
  ) {
    onnx.identityWeightAliases().forEach((from, to) -> {
      Tensor tensor = tensors.get(from);
      if (tensor != null) {
        tensors.putIfAbsent(to, tensor);
      }
      ConvLayout layout = layouts.get(from);
      if (layout != null) {
        layouts.putIfAbsent(to, layout);
      }
    });
  }

  void remapWeights(final Map<String, Tensor> tensors, final Map<String, ConvLayout> layouts) {
    requireNonNull(tensors, "tensors");
    requireNonNull(layouts, "layouts");
  }

  @Override
  public abstract TextToSpeech createSynthesis(Config.HfConfig config, WeightBag weights);
}
