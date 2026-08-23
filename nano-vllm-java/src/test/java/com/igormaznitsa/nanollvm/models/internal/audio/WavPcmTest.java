package com.igormaznitsa.nanollvm.models.internal.audio;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.testsupport.TestWavs;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class WavPcmTest {

  @Test
  void readsPcm16Mono(@TempDir final Path dir) throws Exception {
    Path wav = dir.resolve("tone.wav");
    Files.write(wav, TestWavs.pcm16Mono(new float[] {0.5f, -0.5f, 0f}, 16_000));

    WavPcm.MonoPcm pcm = WavPcm.read(wav);

    assertEquals(16_000, pcm.sampleRate());
    assertEquals(3, pcm.samples().length);
    assertEquals(0.5f, pcm.samples()[0], 1e-4f);
    assertEquals(-0.5f, pcm.samples()[1], 1e-4f);
    assertEquals(0f, pcm.samples()[2], 1e-4f);
  }

  @Test
  void mixesStereoToMono(@TempDir final Path dir) throws Exception {
    Path wav = dir.resolve("stereo.wav");
    Files.write(wav, TestWavs.pcm16Stereo(new float[] {1f, 1f}, new float[] {-1f, -1f}, 8_000));

    WavPcm.MonoPcm pcm = WavPcm.read(wav);

    assertEquals(8_000, pcm.sampleRate());
    assertArrayEquals(new float[] {0f, 0f}, pcm.samples(), 1e-4f);
  }

  @Test
  void readsIeeeFloat32(@TempDir final Path dir) throws Exception {
    Path wav = dir.resolve("float.wav");
    Files.write(wav, TestWavs.float32Mono(new float[] {0.25f, -0.25f}, 16_000));

    WavPcm.MonoPcm pcm = WavPcm.read(wav);

    assertEquals(16_000, pcm.sampleRate());
    assertArrayEquals(new float[] {0.25f, -0.25f}, pcm.samples(), 1e-6f);
  }

  @Test
  void readsBundledCallWav() throws Exception {
    WavPcm.MonoPcm pcm = WavPcm.read(TestWavs.classpathFile("wav/call1.wav"));

    assertEquals(48_000, pcm.sampleRate());
    assertEquals(303_000, pcm.samples().length);
    float peak = 0f;
    for (float sample : pcm.samples()) {
      peak = Math.max(peak, Math.abs(sample));
    }
    assertTrue(peak > 0.05f);
  }

  @Test
  void rejectsCompressedOrTruncatedFiles(@TempDir final Path dir) throws Exception {
    Path truncated = dir.resolve("short.wav");
    Files.write(truncated, new byte[] {1, 2, 3, 4});
    assertThrows(IllegalArgumentException.class, () -> WavPcm.read(truncated));

    Path compressed = dir.resolve("adpcm.wav");
    byte[] bytes = TestWavs.pcm16Mono(new float[] {0f}, 8_000);
    bytes[20] = 7;
    bytes[21] = 0;
    Files.write(compressed, bytes);
    IllegalArgumentException ex =
      assertThrows(IllegalArgumentException.class, () -> WavPcm.read(compressed));
    assertTrue(ex.getMessage().contains("not supported"));
  }
}
