package com.igormaznitsa.nanollvm.weather;

import static java.util.Objects.requireNonNull;

import java.io.ByteArrayInputStream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;

/**
 * One WAV on the default sound line.
 *
 * <p>{@code Clip} cannot play this. Its device buffer is one second, and a clip
 * longer than that but shorter than two seconds stops at the first buffer:
 * "I am looking for location." is about 1.7 s and is heard as "I am".
 * Prefixing "so " pushes the same sentence past two seconds, so that clip plays
 * through. The line below is given the whole PCM buffer and {@code drain()}
 * waits until the device has played it.
 */
public final class SoundClip {

  private final byte[] wavData;

  public SoundClip(final byte[] wav) {
    this.wavData = requireNonNull(wav, "wav").clone();
  }

  public void play() {
    if (this.wavData.length == 0) {
      return;
    }
    try (AudioInputStream stream =
           AudioSystem.getAudioInputStream(new ByteArrayInputStream(this.wavData))) {
      AudioFormat format = stream.getFormat();
      byte[] pcm = stream.readAllBytes();
      if (pcm.length == 0) {
        return;
      }
      SourceDataLine line = AudioSystem.getSourceDataLine(format);
      line.open(format, this.bufferBytes(format, pcm.length));
      try {
        line.start();
        this.writeAll(line, pcm);
        line.drain();
      } finally {
        line.stop();
        line.close();
      }
    } catch (Exception ex) {
      throw new IllegalStateException("could not open the default sound line", ex);
    }
  }

  private int bufferBytes(final AudioFormat format, final int pcmBytes) {
    int frame = format.getFrameSize();
    if (frame <= 0) {
      return pcmBytes;
    }
    return Math.max(pcmBytes, frame * 32);
  }

  private void writeAll(final SourceDataLine line, final byte[] pcm) {
    int offset = 0;
    while (offset < pcm.length) {
      int written = line.write(pcm, offset, pcm.length - offset);
      if (written <= 0) {
        break;
      }
      offset += written;
    }
  }
}
