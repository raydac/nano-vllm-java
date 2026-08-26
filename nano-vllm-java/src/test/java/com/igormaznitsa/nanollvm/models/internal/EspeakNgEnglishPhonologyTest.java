package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

final class EspeakNgEnglishPhonologyTest {

  @Test
  void mapsLotAndRhoticTowardEnUs() {
    assertEquals("ðə", EspeakNgEnglishPhonology.apply("ðə"));
    assertEquals("kwˈɪk", EspeakNgEnglishPhonology.apply("kwɪk"));
    assertEquals("bɹˈaʊn", EspeakNgEnglishPhonology.apply("braʊən"));
    assertEquals("fˈɑːks", EspeakNgEnglishPhonology.apply("fɒks"));
    assertEquals("ˈoʊvɚ", EspeakNgEnglishPhonology.apply("oʊvɜ"));
    assertEquals("dˈɑːɡ", EspeakNgEnglishPhonology.apply("dɒɡ"));
  }

  @Test
  void keepsExistingStress() {
    assertEquals("həlˈoʊ", EspeakNgEnglishPhonology.apply("həlˈoʊ"));
  }

  @Test
  void skipsInitialSchwaWhenPlacingStress() {
    assertFalse(EspeakNgEnglishPhonology.apply("ðə").contains("ˈ"));
    assertEquals("əbˈaʊt", EspeakNgEnglishPhonology.apply("əbaʊt"));
  }
}
