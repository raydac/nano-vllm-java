package com.igormaznitsa.nanollvm.models.internal.audio;

import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Uncompressed WAV reader: PCM integer (8/16/32-bit) or IEEE float32, mixed to mono.
 *
 * @since 1.3.0
 */
public final class WavPcm {

  private WavPcm() {
  }

  /**
   * Reads an uncompressed WAV file into mono float samples in {@code [-1, 1]} plus sample rate.
   *
   * @param wav path to a {@code .wav} file
   * @return mono PCM and original sample rate
   * @throws IOException              if the file cannot be read
   * @throws IllegalArgumentException if the container is compressed or malformed
   */
  public static MonoPcm read(final Path wav) throws IOException {
    Path file = requireNonNull(wav, "wav").toAbsolutePath().normalize();
    return WavPcm.parse(Files.readAllBytes(file), file.toString());
  }

  /**
   * Parses an uncompressed WAV payload in memory (same container as {@link #read(Path)}).
   *
   * @param wav RIFF/WAVE bytes; must not be {@code null}
   * @return mono PCM and original sample rate
   * @throws IllegalArgumentException if the container is compressed or malformed
   */
  public static MonoPcm read(final byte[] wav) {
    return WavPcm.parse(requireNonNull(wav, "wav"), "WAV bytes");
  }

  private static MonoPcm parse(final byte[] bytes, final String source) {
    if (bytes.length < 44) {
      throw new IllegalArgumentException("WAV too short: " + source);
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    if (buf.getInt() != 0x46464952) {
      throw new IllegalArgumentException("not a RIFF WAV: " + source);
    }
    buf.getInt();
    if (buf.getInt() != 0x45564157) {
      throw new IllegalArgumentException("not a WAVE file: " + source);
    }
    int audioFormat = -1;
    int channels = 0;
    int sampleRate = 0;
    int bitsPerSample = 0;
    int dataOffset = -1;
    int dataLength = 0;
    while (buf.remaining() >= 8) {
      int chunkId = buf.getInt();
      int chunkSize = buf.getInt();
      if (chunkSize < 0 || chunkSize > buf.remaining()) {
        throw new IllegalArgumentException("truncated WAV chunk in " + source);
      }
      int chunkStart = buf.position();
      if (chunkId == 0x20746d66) {
        audioFormat = buf.getShort() & 0xffff;
        channels = buf.getShort() & 0xffff;
        sampleRate = buf.getInt();
        buf.getInt();
        buf.getShort();
        bitsPerSample = buf.getShort() & 0xffff;
      } else if (chunkId == 0x61746164) {
        dataOffset = chunkStart;
        dataLength = chunkSize;
      }
      int next = chunkStart + chunkSize + (chunkSize & 1);
      if (next > bytes.length) {
        break;
      }
      buf.position(next);
    }
    if (dataOffset < 0 || channels < 1 || sampleRate < 1 || bitsPerSample < 8) {
      throw new IllegalArgumentException("WAV missing fmt/data: " + source);
    }
    if (audioFormat != 1 && audioFormat != 3) {
      throw new IllegalArgumentException(
        "WAV compression %d is not supported (need PCM or IEEE float): %s"
          .formatted(audioFormat, source));
    }
    float[] interleaved = decodePcm(bytes, dataOffset, dataLength, audioFormat, bitsPerSample);
    return new MonoPcm(mixMono(interleaved, channels), sampleRate);
  }

  private static float[] decodePcm(
    final byte[] bytes,
    final int offset,
    final int length,
    final int audioFormat,
    final int bitsPerSample
  ) {
    ByteBuffer pcm = ByteBuffer.wrap(bytes, offset, length).order(ByteOrder.LITTLE_ENDIAN);
    if (audioFormat == 3) {
      if (bitsPerSample != 32) {
        throw new IllegalArgumentException("IEEE float WAV must be 32-bit");
      }
      int n = length / 4;
      float[] out = new float[n];
      for (int i = 0; i < n; i++) {
        out[i] = pcm.getFloat();
      }
      return out;
    }
    int bytesPerSample = bitsPerSample / 8;
    int n = length / bytesPerSample;
    float[] out = new float[n];
    float scale = (float) (1 << (bitsPerSample - 1));
    for (int i = 0; i < n; i++) {
      int sample = switch (bitsPerSample) {
        case 8 -> (pcm.get() & 0xff) - 128;
        case 16 -> pcm.getShort();
        case 32 -> pcm.getInt();
        default -> throw new IllegalArgumentException(
          "PCM bit depth %d is not supported".formatted(bitsPerSample));
      };
      out[i] = sample / scale;
    }
    return out;
  }

  private static float[] mixMono(final float[] interleaved, final int channels) {
    if (channels == 1) {
      return interleaved;
    }
    int frames = interleaved.length / channels;
    float[] mono = new float[frames];
    float inv = 1f / channels;
    for (int i = 0; i < frames; i++) {
      float sum = 0f;
      int base = i * channels;
      for (int c = 0; c < channels; c++) {
        sum += interleaved[base + c];
      }
      mono[i] = sum * inv;
    }
    return mono;
  }

  /**
   * Encodes mono float samples in {@code [-1, 1]} as an uncompressed PCM16 little-endian WAV.
   *
   * @param samples    mono PCM; must not be {@code null}
   * @param sampleRate Hertz; must be {@code >= 1}
   * @return RIFF/WAVE bytes (44-byte header plus PCM16 frames)
   * @throws NullPointerException     if {@code samples} is {@code null}
   * @throws IllegalArgumentException if {@code sampleRate < 1}
   * @since 1.3.0
   */
  public static byte[] toWav16Le(final float[] samples, final int sampleRate) {
    requireNonNull(samples, "samples");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    int dataBytes = samples.length * 2;
    ByteBuffer wav = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
    wav.putInt(0x46464952);
    wav.putInt(36 + dataBytes);
    wav.putInt(0x45564157);
    wav.putInt(0x20746d66);
    wav.putInt(16);
    wav.putShort((short) 1);
    wav.putShort((short) 1);
    wav.putInt(sampleRate);
    wav.putInt(sampleRate * 2);
    wav.putShort((short) 2);
    wav.putShort((short) 16);
    wav.putInt(0x61746164);
    wav.putInt(dataBytes);
    for (float sample : samples) {
      float clipped = Math.clamp(sample, -1f, 1f);
      wav.putShort((short) Math.round(clipped * 32767f));
    }
    return wav.array();
  }

  /**
   * Mono PCM plus the file's sample rate.
   *
   * @param samples    float samples in {@code [-1, 1]}
   * @param sampleRate Hertz
   */
  @SuppressWarnings("ArrayRecordComponent")
  public record MonoPcm(float[] samples, int sampleRate) {
    public MonoPcm {
      requireNonNull(samples, "samples");
      if (sampleRate < 1) {
        throw new IllegalArgumentException("sampleRate must be >= 1");
      }
      samples = samples.clone();
    }

    @Override
    public float[] samples() {
      return this.samples.clone();
    }
  }
}
