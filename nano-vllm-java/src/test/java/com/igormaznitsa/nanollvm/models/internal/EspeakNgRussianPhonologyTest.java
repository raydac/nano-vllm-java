package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

final class EspeakNgRussianPhonologyTest {

  @Test
  void palatalizesBeforeIAndEAndGuessesFirstSyllableOnTwoSyllableStopFinal() {
    assertEquals("pr;'iv;it", EspeakNgRussianPhonology.apply("privet"));
  }

  @Test
  void reducesUnstressedO() {
    assertEquals("m'olVkV", EspeakNgRussianPhonology.apply("moloko"));
  }

  @Test
  void lexicalStressMarksRequestedSyllable() {
    assertEquals("mVlVk'o", EspeakNgRussianPhonology.apply("moloko", 3));
  }

  @Test
  void twoSyllableVowelFinalGuessesFirstSyllable() {
    assertEquals("'etV", EspeakNgRussianPhonology.apply("eto"));
  }

  @Test
  void doesNotPalatalizeHardHushing() {
    assertEquals("S'i", EspeakNgRussianPhonology.apply("Si"));
  }
}
