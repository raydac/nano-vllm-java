package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_PIPER;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.models.internal.PiperForTts;
import com.igormaznitsa.nanollvm.models.internal.TextToSpeech;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Map;

/**
 * Piper VITS text-to-speech from a voice {@code *.onnx} plus {@code *.onnx.json} sidecar.
 *
 * @since 1.3.0
 */
final class PiperProcessor extends SynthesisArchitecture {

  static final PiperProcessor INSTANCE = new PiperProcessor();

  private PiperProcessor() {
  }

  @Override
  public String architectureId() {
    return ARCH_PIPER;
  }

  @Override
  void remapWeights(final Map<String, Tensor> tensors, final Map<String, ConvLayout> layouts) {
    PiperOnnxNames.promoteDecoderResblocks(tensors, layouts);
  }

  @Override
  public TextToSpeech createSynthesis(final Config.HfConfig config, final WeightBag weights) {
    return new PiperForTts(config, weights);
  }
}
