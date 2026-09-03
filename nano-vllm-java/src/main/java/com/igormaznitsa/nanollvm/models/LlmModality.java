package com.igormaznitsa.nanollvm.models;

import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.ofNullable;

import java.util.Locale;
import java.util.Optional;

/**
 * Content type a loaded {@link LlmModel} can accept or produce.
 *
 * <p>Chat graphs in this library currently <em>run</em> {@link #TEXT} on both sides. Embedding
 * encoders accept {@link #TEXT} and produce {@link #EMBEDDING}. {@link #IMAGE}, {@link #AUDIO},
 * and {@link #VIDEO} appear on {@link LlmModel#modalities()} when the checkpoint declares those
 * towers (Gemma 4 QAT mobile does). Whisper speech graphs run {@link #AUDIO} in and
 * {@link #TEXT} out. fastText classifiers run {@link #TEXT} in and {@link #LABELS} out.
 * {@link LlmModel#usableModalities()} stays text-only for chat checkpoints until
 * vision/audio towers are wired.
 *
 * <p>Constants used as {@code generate} outputs carry a {@link #resultType()}: {@link #TEXT} →
 * {@link LlmOutText}, {@link #AUDIO} → {@link LlmOutSoundData}, {@link #EMBEDDING} →
 * {@link LlmOutEmbedding}, {@link #LABELS} → {@link LlmOutLabels}. {@link #IMAGE} and
 * {@link #VIDEO} are declaration-only until those towers run.
 *
 * @since 1.2.0
 */
public enum LlmModality {

  /**
   * Token / string content (chat, completion, embedding input; Whisper / completion output).
   */
  TEXT("text", LlmOutText.class),
  /**
   * Pixel / image content.
   */
  IMAGE("image", null),
  /**
   * Waveform / audio content (Whisper input; Piper output).
   */
  AUDIO("audio", LlmOutSoundData.class),
  /**
   * Moving-image / video content.
   */
  VIDEO("video", null),
  /**
   * Dense vector from {@link LlmModel#generate(LlmInput, LlmModality)}.
   */
  EMBEDDING("embedding", LlmOutEmbedding.class),
  /**
   * Ranked classification labels from {@link LlmModel#generate(LlmInput, LlmModality)}
   * (fastText language id and other supervised classifiers).
   *
   * @since 1.4.0
   */
  LABELS("labels", LlmOutLabels.class);

  private final String wireName;
  private final Class<? extends LlmOutput> resultType;

  LlmModality(final String wireName, final Class<? extends LlmOutput> resultType) {
    this.wireName = requireNonNull(wireName, "wireName");
    this.resultType = resultType;
  }

  /**
   * Parses a lowercase wire name ({@code text}, {@code image}, {@code audio}, {@code video},
   * {@code embedding}, {@code labels}). Blank or unknown names are empty.
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

  /**
   * Concrete {@link LlmOutput} type produced when this constant is used as a {@code generate}
   * output modality. Empty for declaration-only towers ({@link #IMAGE}, {@link #VIDEO}).
   *
   * @since 1.4.0
   */
  public Optional<Class<? extends LlmOutput>> resultType() {
    return ofNullable(this.resultType);
  }

  /**
   * Casts {@code output} to this modality's {@link #resultType()}.
   *
   * @param output result to cast; must not be {@code null}
   * @param <T>    expected {@link LlmOutput} subtype (inferred from the assignment)
   * @return {@code output} as {@code T}
   * @throws NullPointerException  if {@code output} is {@code null}
   * @throws IllegalStateException if this constant has no result type
   * @throws ClassCastException    if {@code output} is not an instance of the result type
   * @since 1.4.0
   */
  @SuppressWarnings({"unchecked", "TypeParameterUnusedInFormals"})
  public <T extends LlmOutput> T cast(final LlmOutput output) {
    requireNonNull(output, "output");
    if (this.resultType == null) {
      throw new IllegalStateException(
        "'%s' is not a generate result modality".formatted(this.wireName));
    }
    return (T) this.resultType.cast(output);
  }
}
