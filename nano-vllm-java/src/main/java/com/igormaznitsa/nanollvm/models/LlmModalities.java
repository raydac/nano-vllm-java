package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.Config;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Input and output {@link LlmModality} sets for a loaded {@link LlmModel}.
 *
 * <p>Chat graphs declare at least {@link LlmModality#TEXT} in and out. Embedding encoders are
 * {@link #TEXT_TO_EMBEDDING}. Extra input types ({@link LlmModality#IMAGE}, {@link LlmModality#AUDIO},
 * {@link LlmModality#VIDEO}) come from checkpoint keys such as {@code vision_config} /
 * {@code audio_config} / {@code video_token_id}. This library currently <em>runs</em>
 * {@link #TEXT_TO_TEXT}, {@link #TEXT_TO_EMBEDDING}, or {@link #AUDIO_TO_TEXT} — see
 * {@link LlmModel#usableModalities()}.
 *
 * @param input  content types the checkpoint consumes; never empty
 * @param output content types the checkpoint produces; may be empty
 * @since 1.2.0
 */
public record LlmModalities(Set<LlmModality> input, Set<LlmModality> output) {

  /**
   * Causal chat / completion: text in, text out.
   */
  public static final LlmModalities TEXT_TO_TEXT = of(LlmModality.TEXT, LlmModality.TEXT);

  /**
   * Embedding encoder: text in, vector out.
   */
  public static final LlmModalities TEXT_TO_EMBEDDING = of(LlmModality.TEXT, LlmModality.EMBEDDING);

  /**
   * Whisper speech-to-text: audio in, text out.
   *
   * @since 1.3.0
   */
  public static final LlmModalities AUDIO_TO_TEXT = of(LlmModality.AUDIO, LlmModality.TEXT);

  /**
   * Canonical constructor: copies both sides, rejects a null or empty input set.
   *
   * @throws NullPointerException     if a set or an element is {@code null}
   * @throws IllegalArgumentException if {@code input} is empty
   */
  public LlmModalities {
    input = freeze("input", input);
    output = freeze("output", output);
    if (input.isEmpty()) {
      throw new IllegalArgumentException("input modalities must not be empty");
    }
  }

  @Override
  public Set<LlmModality> input() {
    return Set.copyOf(this.input);
  }

  @Override
  public Set<LlmModality> output() {
    return Set.copyOf(this.output);
  }

  /**
   * Single input type and single output type.
   *
   * @param input  consumed modality; never {@code null}
   * @param output produced modality; never {@code null}
   * @return frozen pair
   * @since 1.2.0
   */
  public static LlmModalities of(final LlmModality input, final LlmModality output) {
    return new LlmModalities(
      Set.of(requireNonNull(input, "input")),
      Set.of(requireNonNull(output, "output")));
  }

  /**
   * Arbitrary input and output sets (copied).
   *
   * @param input  consumed modalities; never {@code null} or empty
   * @param output produced modalities; never {@code null} (may be empty)
   * @return frozen pair
   * @since 1.2.0
   */
  public static LlmModalities of(final Set<LlmModality> input, final Set<LlmModality> output) {
    return new LlmModalities(input, output);
  }

  /**
   * Modalities declared by the checkpoint config. Embedding files are
   * {@link #TEXT_TO_EMBEDDING}. Chat files always include text; image / audio / video are added
   * when the corresponding HF keys are present.
   *
   * @param config    parsed Hugging Face / GGUF-mapped config; never {@code null}
   * @param embedding {@code true} for BERT-style encoders
   * @return frozen checkpoint pair
   * @since 1.2.0
   */
  public static LlmModalities ofCheckpoint(final Config.HfConfig config, final boolean embedding) {
    requireNonNull(config, "config");
    if (embedding) {
      return TEXT_TO_EMBEDDING;
    }
    if (config.isWhisper()) {
      return AUDIO_TO_TEXT;
    }
    EnumSet<LlmModality> input = EnumSet.of(LlmModality.TEXT);
    if (config.imageConfigPresent()) {
      input.add(LlmModality.IMAGE);
    }
    if (config.audioConfigPresent()) {
      input.add(LlmModality.AUDIO);
    }
    if (config.videoConfigPresent()) {
      input.add(LlmModality.VIDEO);
    }
    return of(input, Set.of(LlmModality.TEXT));
  }

  /**
   * Modalities this library actually consumes and produces for a loaded graph.
   *
   * @param embedding {@code true} for BERT-style encoders
   * @return {@link #TEXT_TO_EMBEDDING} or {@link #TEXT_TO_TEXT}
   * @since 1.2.0
   */
  public static LlmModalities usable(final boolean embedding) {
    return usable(embedding, false);
  }

  /**
   * Modalities this library actually consumes and produces for a loaded graph.
   *
   * @param embedding {@code true} for BERT-style encoders
   * @param speech    {@code true} for Whisper speech-to-text
   * @return {@link #AUDIO_TO_TEXT}, {@link #TEXT_TO_EMBEDDING}, or {@link #TEXT_TO_TEXT}
   * @since 1.3.0
   */
  public static LlmModalities usable(final boolean embedding, final boolean speech) {
    if (speech) {
      return AUDIO_TO_TEXT;
    }
    return embedding ? TEXT_TO_EMBEDDING : TEXT_TO_TEXT;
  }

  private static Set<LlmModality> freeze(final String role, final Set<LlmModality> modalities) {
    requireNonNull(modalities, role);
    return modalities.isEmpty()
      ? Set.of()
      : Collections.unmodifiableSet(EnumSet.copyOf(modalities));
  }

  private static String format(final Set<LlmModality> side) {
    return side.isEmpty() ? "none" : side.stream().map(LlmModality::wireName).collect(joining("+"));
  }

  /**
   * {@code true} when this checkpoint consumes {@code modality}.
   *
   * @since 1.2.0
   */
  public boolean accepts(final LlmModality modality) {
    return this.input.contains(requireNonNull(modality, "modality"));
  }

  /**
   * {@code true} when this checkpoint produces {@code modality}.
   *
   * @since 1.2.0
   */
  public boolean emits(final LlmModality modality) {
    return this.output.contains(requireNonNull(modality, "modality"));
  }

  /**
   * Wire names joined as {@code text+image->text} (empty output is {@code none}).
   */
  @Override
  public String toString() {
    return format(this.input) + "->" + format(this.output);
  }
}
