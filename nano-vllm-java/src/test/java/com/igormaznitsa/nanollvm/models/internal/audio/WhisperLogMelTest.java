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
}
