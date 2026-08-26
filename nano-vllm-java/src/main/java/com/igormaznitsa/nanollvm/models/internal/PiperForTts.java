package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_PIPER;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * Piper VITS text-to-speech: espeak-ng-data G2P, then a Java generator graph.
 *
 * @since 1.3.0
 */
public final class PiperForTts implements TextToSpeech {

  private final Config.HfConfig config;
  private final VitsSynthesizer vits;
  private EspeakNgG2p g2p;
  private Path g2pDataDir;

  /**
   * Binds VITS weights. {@code config.piper()} must be present.
   *
   * @param config  Piper sidecar blueprint; must not be {@code null}
   * @param weights ONNX initializers; must not be {@code null}
   */
  public PiperForTts(final Config.HfConfig config, final WeightBag weights) {
    this.config = requireNonNull(config, "config");
    if (this.config.piper() == null) {
      throw new IllegalArgumentException("piper spec is required");
    }
    this.vits = new VitsSynthesizer(this.config, requireNonNull(weights, "weights"));
  }

  @Override
  public String architectureName() {
    return ARCH_PIPER;
  }

  @Override
  public int sampleRate() {
    return this.config.piper().sampleRate();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public float[] synthesize(
    final CharSequence text,
    final Path espeakData,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(text, "text");
    requireNonNull(espeakData, "espeakData");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    if (text.toString().isBlank()) {
      throw new IllegalArgumentException("text must not be blank");
    }
    Config.PiperSpec spec = this.config.piper();
    List<Integer> ids = this.g2pFor(espeakData, spec).phonemeIds(text);
    int[] phonemes = new int[ids.size()];
    for (int i = 0; i < ids.size(); i++) {
      phonemes[i] = ids.get(i);
    }
    return this.vits.infer(
      phonemes, spec.noiseScale(), spec.lengthScale(), spec.noiseW(), runtime, random);
  }

  private EspeakNgG2p g2pFor(final Path espeakData, final Config.PiperSpec spec) {
    Path normalized = espeakData.toAbsolutePath().normalize();
    if (this.g2p == null || !normalized.equals(this.g2pDataDir)) {
      this.g2p = new EspeakNgG2p(spec.phonemeIdMap(), normalized, spec.espeakVoice());
      this.g2pDataDir = normalized;
    }
    return this.g2p;
  }
}
