package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Locale;
import org.junit.jupiter.api.Test;

final class WhisperLanguagesTest {

  @Test
  void nullAndRootMeanAutoDetect() {
    assertTrue(WhisperLanguages.tokenCode(null).isEmpty());
    assertTrue(WhisperLanguages.tokenCode(Locale.ROOT).isEmpty());
  }

  @Test
  void usesIsoLanguageAndDropsRegion() {
    assertEquals("en", WhisperLanguages.tokenCode(Locale.ENGLISH).orElseThrow());
    assertEquals("en", WhisperLanguages.tokenCode(Locale.US).orElseThrow());
    assertEquals("de", WhisperLanguages.tokenCode(Locale.GERMANY).orElseThrow());
    assertEquals("zh", WhisperLanguages.tokenCode(Locale.CHINESE).orElseThrow());
    assertEquals("yue", WhisperLanguages.tokenCode(Locale.forLanguageTag("yue")).orElseThrow());
  }

  @Test
  void mapsIsoCodesWhisperStillSpellsTheOldWay() {
    assertEquals("jw", WhisperLanguages.tokenCode(Locale.forLanguageTag("jv")).orElseThrow());
    assertEquals("no", WhisperLanguages.tokenCode(Locale.forLanguageTag("nb")).orElseThrow());
    assertEquals("he", WhisperLanguages.tokenCode(Locale.forLanguageTag("iw")).orElseThrow());
    assertEquals("id", WhisperLanguages.tokenCode(Locale.forLanguageTag("in")).orElseThrow());
    assertEquals("tl", WhisperLanguages.tokenCode(Locale.forLanguageTag("fil")).orElseThrow());
  }
}
