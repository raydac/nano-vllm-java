package com.igormaznitsa.nanollvm.weather.speech;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

class WavSpeakerTest {

  @Test
  void emptyWavIsNotPlayed() {
    assertFalse(WavSpeaker.play(new byte[0]));
  }
}
