package com.igormaznitsa.nanollvm.models;

import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;

import java.util.Locale;
import java.util.Optional;

/**
 * Content type a loaded {@link LlmModel} can accept or produce.
 *
 * <p>Chat graphs in this library currently <em>run</em> {@link #TEXT} on both sides. Embedding
 * encoders accept {@link #TEXT} and produce {@link #EMBEDDING}. {@link #IMAGE}, {@link #AUDIO},
 * and {@link #VIDEO} appear on {@link LlmModel#modalities()} when the checkpoint declares those
 * towers (Gemma 4 QAT mobile does). Whisper speech graphs run {@link #AUDIO} in and
 * {@link #TEXT} out. {@link LlmModel#usableModalities()} stays text-only for chat
 * checkpoints until vision/audio towers are wired.
 *
 * @since 1.2.0
 */
public enum LlmModality {

  /**
   * Token / string content (chat, completion, embedding input).
   */
  TEXT("text"),
  /**
   * Pixel / image content.
   */
  IMAGE("image"),
  /**
   * Waveform / audio content.
   */
  AUDIO("audio"),
  /**
   * Moving-image / video content.
   */
  VIDEO("video"),
  /**
   * Dense vector from {@link LlmModel#generate(LlmInput, LlmModality)} with
   * {@link #EMBEDDING} output.
   */
  EMBEDDING("embedding");

  private final String wireName;

  LlmModality(final String wireName) {
    this.wireName = requireNonNull(wireName, "wireName");
  }

  /**
   * Parses a lowercase wire name ({@code text}, {@code image}, {@code audio}, {@code video},
   * {@code embedding}). Blank or unknown names are empty.
   *
   * @param name wire name; {@code null} or blank → empty
   * @return matching constant, or empty when unrecognized
   * @since 1.2.0
   */
  public static Optional<LlmModality> fromWire(final String name) {
    if (name == null || name.isBlank()) {
      return Optional.empty();
    }
    String key = name.strip().toLowerCase(Locale.ROOT);
    return stream(values()).filter(modality -> modality.wireName().equals(key)).findFirst();
  }

  /**
   * Stable lowercase name for logs and JSON-style APIs.
   *
   * @since 1.2.0
   */
  public String wireName() {
    return this.wireName;
  }
}
