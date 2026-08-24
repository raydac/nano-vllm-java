package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

final class EspeakNgPhonemesTest {

  @Test
  void kirschenbaumHelloBecomesIpa() {
    assertEquals("həlˈoʊ", EspeakNgPhonemes.toIpa("h@l'oU"));
  }

  @Test
  void separatorsAndLengthMarks() {
    assertEquals("tʃiː", EspeakNgPhonemes.toIpa("tSi:"));
    assertEquals("sʲ", EspeakNgPhonemes.toIpa("s;"));
    assertEquals("jɪ", EspeakNgPhonemes.toIpa("jI3"));
  }
}
