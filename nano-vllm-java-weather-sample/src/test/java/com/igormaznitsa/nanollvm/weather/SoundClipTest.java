package com.igormaznitsa.nanollvm.weather;

import org.junit.jupiter.api.Test;

class SoundClipTest {

  @Test
  void emptyWavDoesNotPlay() {
    new SoundClip(new byte[0]).play();
  }
}
