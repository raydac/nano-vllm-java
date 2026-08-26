package com.igormaznitsa.nanollvm.models.internal.audio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

final class WhisperLogMelTest {

  @Test
  void silenceYieldsFiniteWhisperShapedFeatures() {
    Tensor mel =
      WhisperLogMel.features(new float[WhisperLogMel.SAMPLE_RATE], WhisperLogMel.SAMPLE_RATE);

    assertEquals(WhisperLogMel.N_MELS, mel.size(0));
    assertEquals(WhisperLogMel.N_FRAMES, mel.size(1));
    for (int i = 0; i < mel.numel(); i++) {
      assertTrue(Float.isFinite(mel.get(i)), "bin " + i);
    }
  }

  @Test
  void emptyPcmIsPaddedToThirtySeconds() {
    Tensor mel = WhisperLogMel.features(new float[0], WhisperLogMel.SAMPLE_RATE);

    assertEquals(WhisperLogMel.N_MELS, mel.size(0));
    assertEquals(WhisperLogMel.N_FRAMES, mel.size(1));
  }

  @Test
  void sliceFeaturesMatchCopiedChunk() {
    float[] pcm = new float[WhisperLogMel.SAMPLE_RATE * 2];
    for (int i = 0; i < pcm.length; i++) {
      pcm[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / WhisperLogMel.SAMPLE_RATE);
    }
    int origin = WhisperLogMel.SAMPLE_RATE / 4;
    int length = WhisperLogMel.SAMPLE_RATE;
    float[] copy = new float[length];
    System.arraycopy(pcm, origin, copy, 0, length);

    Tensor fromSlice = WhisperLogMel.featuresAt16k(
      pcm, origin, length, com.igormaznitsa.nanollvm.tensor.MatmulRuntime.sequential());
    Tensor fromCopy = WhisperLogMel.features(copy, WhisperLogMel.SAMPLE_RATE);

    assertEquals(fromCopy.numel(), fromSlice.numel());
    for (int i = 0; i < fromCopy.numel(); i++) {
      assertEquals(fromCopy.get(i), fromSlice.get(i), 1e-5f, "bin " + i);
    }
  }
}
