package com.igormaznitsa.nanollvm.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Locale;
import org.junit.jupiter.api.Test;

final class LlmInputOutputTest {

  @Test
  void textInputRejectsBlank() {
    assertThrows(IllegalArgumentException.class, () -> LlmInText.of("  "));
    assertThrows(NullPointerException.class, () -> LlmInText.of(null));
  }

  @Test
  void soundInputIsXorWavOrPcm() {
    LlmInSound wav = LlmInSound.ofWav(new byte[] {1, 2, 3}, Locale.ENGLISH);
    assertTrue(wav.isWav());
    assertFalse(wav.isPcm());
    assertEquals(0, wav.sampleRate());
    assertEquals(Locale.ENGLISH, wav.language());
    assertArrayEquals(new byte[] {1, 2, 3}, wav.wav());
    assertNull(wav.pcm());

    LlmInSound pcm = LlmInSound.ofPcm(new float[] {0.1f, -0.2f}, 16_000);
    assertTrue(pcm.isPcm());
    assertFalse(pcm.isWav());
    assertEquals(16_000, pcm.sampleRate());
    assertNull(pcm.language());
    assertArrayEquals(new float[] {0.1f, -0.2f}, pcm.pcm());
    assertNull(pcm.wav());

    assertThrows(IllegalArgumentException.class, () -> new LlmInSound(null, null, 0, null));
    assertThrows(
      IllegalArgumentException.class,
      () -> new LlmInSound(new byte[] {1}, new float[] {0f}, 16_000, null));
    assertThrows(IllegalArgumentException.class, () -> LlmInSound.ofPcm(new float[] {0f}, 0));
  }

  @Test
  void soundInputDefensiveCopies() {
    byte[] raw = {9, 8, 7};
    LlmInSound wav = LlmInSound.ofWav(raw);
    raw[0] = 0;
    assertEquals(9, wav.wav()[0]);

    float[] samples = {1f};
    LlmInSound pcm = LlmInSound.ofPcm(samples, 8_000);
    samples[0] = 0f;
    assertEquals(1f, pcm.pcm()[0]);
  }

  @Test
  void outputModalitiesMatchRecords() {
    assertEquals(LlmModality.TEXT, new LlmOutText("hi").modality());
    assertEquals(LlmModality.AUDIO, new LlmOutSoundData(new byte[] {1}, 22_050).modality());
    assertEquals(LlmModality.EMBEDDING, new LlmOutEmbedding(new float[] {0.5f}).modality());

    assertThrows(IllegalArgumentException.class, () -> new LlmOutSoundData(new byte[] {1}, 0));
    assertThrows(IllegalArgumentException.class, () -> new LlmOutEmbedding(new float[0]));
  }
}
