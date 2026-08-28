package com.igormaznitsa.nanollvm.samples;

import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.util.stream.Collectors.toSet;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.LabeledText;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import org.junit.jupiter.api.Test;

class VoiceReplyHelloWorldTest {

  private static boolean containsCyrillic(final String text) {
    return text.codePoints().anyMatch(
      code -> Character.UnicodeBlock.CYRILLIC.equals(Character.UnicodeBlock.of(code)));
  }

  @Test
  void englishMoodCatalogHasFourLabels() {
    List<LabeledText> examples = VoiceReplyHelloWorld.moodExamples(true);
    assertEquals(Set.of("happy", "sad", "angry", "calm"),
      examples.stream().map(LabeledText::label).collect(toSet()));
    assertTrue(examples.size() >= 8);
  }

  @Test
  void russianMoodCatalogMatchesEnglishLabels() {
    assertEquals(
      VoiceReplyHelloWorld.moodExamples(true).stream().map(LabeledText::label).collect(toSet()),
      VoiceReplyHelloWorld.moodExamples(false).stream().map(LabeledText::label).collect(toSet()));
  }

  @Test
  void cannedSpeechFollowsVoiceLanguage() {
    assertFalse(containsCyrillic(VoiceReplyHelloWorld.cannedUserSpeech(true)));
    assertTrue(containsCyrillic(VoiceReplyHelloWorld.cannedUserSpeech(false)));
  }

  @Test
  void chatTurnCarriesMoodAndTranscript() {
    String english = VoiceReplyHelloWorld.chatUserTurn("happy", "we shipped it", true);
    assertTrue(english.contains("happy"));
    assertTrue(english.contains("we shipped it"));

    String russian = VoiceReplyHelloWorld.chatUserTurn("calm", "встреча в три", false);
    assertTrue(russian.contains("calm"));
    assertTrue(russian.contains("встреча в три"));
    assertTrue(containsCyrillic(russian));
  }

  @Test
  void piperFolderLanguageFromName() {
    assertTrue(BundledModels.isEnglishPiperVoice(Path.of("models/piper-en-lessac-medium")));
    assertFalse(BundledModels.isEnglishPiperVoice(Path.of("models/piper-ru-irina-medium")));
  }

  @Test
  void e5PrefixFromFolderName() {
    assertTrue(BundledModels.usesE5QueryPrefix(Path.of("models/multilingual-e5-small")));
    assertFalse(BundledModels.usesE5QueryPrefix(Path.of("models/gte-small.Q2_K.gguf")));
  }

  @Test
  void playWavSkipsEmptyAndInvalidBytes() {
    assertFalse(VoiceReplyHelloWorld.playWav(new byte[0]));
    assertFalse(VoiceReplyHelloWorld.playWav("not a wav".getBytes(US_ASCII)));
  }

  @Test
  void recordDefaultInputSkipsNonPositiveDuration() {
    assertTrue(VoiceReplyHelloWorld.recordDefaultInput(0).isEmpty());
    assertTrue(VoiceReplyHelloWorld.recordDefaultInput(-1).isEmpty());
  }

  @Test
  void downmixStereoPcm16AveragesChannels() {
    ByteBuffer stereo = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    stereo.putShort((short) 1_000);
    stereo.putShort((short) 2_000);
    stereo.putShort((short) -100);
    stereo.putShort((short) 100);

    byte[] mono = VoiceReplyHelloWorld.downmixStereoPcm16(stereo.array());
    ByteBuffer decoded = ByteBuffer.wrap(mono).order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(1_500, decoded.getShort());
    assertEquals(0, decoded.getShort());
  }

  @Test
  void trimSilencePcm16DropsQuietOnlyClips() {
    assertEquals(0, VoiceReplyHelloWorld.trimSilencePcm16(new byte[2_000], 16_000).length);
  }

  @Test
  void trimSilencePcm16KeepsSpeechWithPad() {
    byte[] pcm = new byte[10_100 * 2];
    ByteBuffer buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
    buf.position(5_000 * 2);
    for (int i = 0; i < 100; i++) {
      buf.putShort((short) 8_000);
    }

    byte[] trimmed = VoiceReplyHelloWorld.trimSilencePcm16(pcm, 16_000);
    assertEquals(6_500 * 2, trimmed.length);
    ByteBuffer decoded = ByteBuffer.wrap(trimmed).order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(0, decoded.getShort(0));
    assertEquals(8_000, decoded.getShort(3_200 * 2));
  }

  @Test
  void piperShapedWavIsJavaSoundReadable() throws Exception {
    byte[] wav = VoiceReplyHelloWorld.pcm16MonoWav(22_050, new byte[16]);
    try (AudioInputStream stream = AudioSystem.getAudioInputStream(new ByteArrayInputStream(wav))) {
      assertEquals(22_050f, stream.getFormat().getSampleRate());
      assertEquals(16, stream.getFormat().getSampleSizeInBits());
      assertEquals(1, stream.getFormat().getChannels());
    }
    assertDoesNotThrow(() -> VoiceReplyHelloWorld.playWav(wav));
  }
}
