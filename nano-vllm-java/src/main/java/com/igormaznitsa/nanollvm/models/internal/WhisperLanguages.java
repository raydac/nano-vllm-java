package com.igormaznitsa.nanollvm.models.internal;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Maps a {@link Locale} onto Whisper's {@code <|xx|>} language token. Region is ignored.
 * {@code null} and {@link Locale#ROOT} mean auto-detect.
 */
final class WhisperLanguages {

  private static final Map<String, String> ALIASES = Map.of(
    "jv", "jw",
    "nb", "no",
    "iw", "he",
    "in", "id",
    "ji", "yi",
    "fil", "tl",
    "cmn", "zh"
  );

  private WhisperLanguages() {
  }

  static Optional<String> tokenCode(final Locale language) {
    if (language == null) {
      return Optional.empty();
    }
    String iso = language.getLanguage();
    if (iso.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(ALIASES.getOrDefault(iso, iso));
  }
}
