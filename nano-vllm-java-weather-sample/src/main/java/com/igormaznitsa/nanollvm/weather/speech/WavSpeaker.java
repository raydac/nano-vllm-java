package com.igormaznitsa.nanollvm.weather.speech;

import static java.util.Objects.requireNonNull;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineEvent;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Plays a WAV through the default Java Sound mixer.
 */
public final class WavSpeaker {

  private WavSpeaker() {
  }

  public static boolean play(final byte[] wav) {
    requireNonNull(wav, "wav");
    if (wav.length == 0) {
      return false;
    }
    try (AudioInputStream encoded = AudioSystem.getAudioInputStream(new ByteArrayInputStream(wav));
         AudioInputStream pcm = playable(encoded);
         Clip clip = (Clip) AudioSystem.getLine(new DataLine.Info(Clip.class, pcm.getFormat()))) {
      return playClip(clip, pcm);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      return false;
    } catch (UnsupportedAudioFileException | LineUnavailableException | IOException
             | IllegalArgumentException | SecurityException ex) {
      return false;
    }
  }

  private static boolean playClip(final Clip clip, final AudioInputStream pcm)
    throws IOException, LineUnavailableException, InterruptedException {
    CountDownLatch finished = new CountDownLatch(1);
    clip.addLineListener(event -> {
      if (LineEvent.Type.STOP.equals(event.getType())
        || LineEvent.Type.CLOSE.equals(event.getType())) {
        finished.countDown();
      }
    });
    clip.open(pcm);
    if (clip.getFrameLength() <= 0) {
      return false;
    }

    clip.start();
    if (!finished.await(deadlineMillis(clip), MILLISECONDS)) {
      clip.stop();
    }
    return true;
  }

  private static AudioInputStream playable(final AudioInputStream encoded) {
    AudioFormat source = encoded.getFormat();
    AudioFormat same = pcm16(source.getSampleRate(), Math.max(1, source.getChannels()));
    if (source.matches(same) && lineSupports(same)) {
      return encoded;
    }
    if (AudioSystem.isConversionSupported(same, source) && lineSupports(same)) {
      return AudioSystem.getAudioInputStream(same, encoded);
    }
    AudioFormat mono = pcm16(source.getSampleRate(), 1);
    if (AudioSystem.isConversionSupported(mono, source) && lineSupports(mono)) {
      return AudioSystem.getAudioInputStream(mono, encoded);
    }
    return encoded;
  }

  private static boolean lineSupports(final AudioFormat format) {
    return AudioSystem.isLineSupported(new DataLine.Info(Clip.class, format));
  }

  private static AudioFormat pcm16(final float sampleRate, final int channels) {
    int count = Math.max(1, channels);
    return new AudioFormat(
      AudioFormat.Encoding.PCM_SIGNED, sampleRate, 16, count, count * 2, sampleRate, false);
  }

  private static long deadlineMillis(final Clip clip) {
    int frames = clip.getFrameLength();
    float rate = clip.getFormat().getFrameRate();
    if (frames <= 0 || !Float.isFinite(rate) || rate <= 0f) {
      return 5_000L;
    }
    return Math.min(120_000L, (long) (frames / rate * 1_000.0d) + 2_000L);
  }
}
