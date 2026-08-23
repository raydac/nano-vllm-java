package com.igormaznitsa.nanollvm.testsupport;

import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.util.Objects.requireNonNull;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

public final class TestWavs {

  private TestWavs() {
  }

  public static Path classpathFile(final String resourcePath) {
    String name = resourcePath.startsWith("/") ? resourcePath : "/" + resourcePath;
    URL url =
      requireNonNull(TestWavs.class.getResource(name), () -> "missing test resource " + name);
    try {
      return Path.of(url.toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  public static byte[] pcm16Mono(final float[] samples, final int sampleRate) {
    return pcm(samples, 1, sampleRate, 1, 16);
  }

  public static byte[] pcm16Stereo(final float[] left, final float[] right, final int sampleRate) {
    if (left.length != right.length) {
      throw new IllegalArgumentException("left/right lengths must match");
    }
    float[] interleaved = new float[left.length * 2];
    for (int i = 0; i < left.length; i++) {
      interleaved[i * 2] = left[i];
      interleaved[i * 2 + 1] = right[i];
    }
    return pcm(interleaved, 2, sampleRate, 1, 16);
  }

  public static byte[] float32Mono(final float[] samples, final int sampleRate) {
    return pcm(samples, 1, sampleRate, 3, 32);
  }

  private static byte[] pcm(
    final float[] samples,
    final int channels,
    final int sampleRate,
    final int audioFormat,
    final int bitsPerSample
  ) {
    int bytesPerSample = bitsPerSample / 8;
    int dataLen = samples.length * bytesPerSample;
    ByteBuffer buf = ByteBuffer.allocate(44 + dataLen).order(ByteOrder.LITTLE_ENDIAN);
    buf.put("RIFF".getBytes(US_ASCII));
    buf.putInt(36 + dataLen);
    buf.put("WAVE".getBytes(US_ASCII));
    buf.put("fmt ".getBytes(US_ASCII));
    buf.putInt(16);
    buf.putShort((short) audioFormat);
    buf.putShort((short) channels);
    buf.putInt(sampleRate);
    buf.putInt(sampleRate * channels * bytesPerSample);
    buf.putShort((short) (channels * bytesPerSample));
    buf.putShort((short) bitsPerSample);
    buf.put("data".getBytes(US_ASCII));
    buf.putInt(dataLen);
    if (audioFormat == 3) {
      for (float sample : samples) {
        buf.putFloat(sample);
      }
    } else {
      float scale = (float) (1 << (bitsPerSample - 1));
      for (float sample : samples) {
        int v = Math.round(Math.clamp(sample, -1f, 1f) * (scale - 1f));
        buf.putShort((short) v);
      }
    }
    return buf.array();
  }
}
