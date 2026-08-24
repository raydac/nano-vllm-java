package com.igormaznitsa.nanollvm.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

final class LlmOptionalDataTest {

  @Test
  void espeakKeyCastsPathAndString() {
    Path dir = Path.of("espeak-ng-data").toAbsolutePath().normalize();
    assertEquals(dir, LlmOptionalData.cast(LlmOptionalData.ESPEAK_DATA, dir));
    assertEquals(dir, LlmOptionalData.cast(LlmOptionalData.ESPEAK_DATA, dir.toString()));
    assertEquals("espeak.data", LlmOptionalData.ESPEAK_DATA.id());
  }

  @Test
  void asPathRejectsBlankAndNonPath() {
    assertThrows(IllegalArgumentException.class, () -> LlmOptionalData.asPath("  "));
    assertThrows(IllegalArgumentException.class, () -> LlmOptionalData.asPath(12));
    assertThrows(NullPointerException.class, () -> LlmOptionalData.asPath(null));
  }

  @Test
  void castRejectsWrongTypeAndBlankKey() {
    LlmOptionalData.Key<String> note = LlmOptionalData.Key.of("app.note", String.class);
    assertEquals("ok", LlmOptionalData.cast(note, "ok"));
    assertThrows(IllegalArgumentException.class, () -> LlmOptionalData.cast(note, 1));
    assertThrows(IllegalArgumentException.class, () -> LlmOptionalData.Key.of(" ", String.class));
  }
}
