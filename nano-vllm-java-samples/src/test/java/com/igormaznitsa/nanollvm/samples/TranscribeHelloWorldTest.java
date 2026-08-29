package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class TranscribeHelloWorldTest {

  @Test
  void wavPathUsesCliArgument() {
    Path expected = Path.of("clip.wav").toAbsolutePath().normalize();
    assertEquals(Optional.of(expected), TranscribeHelloWorld.wavPath(new String[] {
      "models/whisper-base",
      "clip.wav"
    }));
  }

  @Test
  void wavPathWithoutArgUsesSynthesizeOutputWhenPresent() {
    Path synthesized = SynthesizeHelloWorld.defaultWavPath();
    Optional<Path> wav = TranscribeHelloWorld.wavPath(new String[] {"models/whisper-base"});
    assertEquals(Files.isRegularFile(synthesized) ? Optional.of(synthesized) : Optional.empty(),
      wav);
  }

  @Test
  void defaultSynthesizeWavIsPiperHello() {
    assertTrue(SynthesizeHelloWorld.defaultWavPath().endsWith("piper-hello.wav"));
  }
}
