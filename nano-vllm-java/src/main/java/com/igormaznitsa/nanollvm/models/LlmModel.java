package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.LlmModelImpl;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Loaded model: architecture weights, HF/GGUF config, and tokenizer.
 *
 * <p>Safe to share across threads and across many {@link LLM} instances (causal models). Mutable
 * inference state (KV cache, scheduler, sampling) lives on each {@link LLM}, not here. Load-time
 * options ({@link #options()}, including {@link #OPTION_THINK_TAGS} and
 * {@link #OPTION_CHAT_SPECIALS}) are frozen at
 * {@link LlmModelFactory#make} and never change. Embedding models expose
 * {@link #embed(CharSequence)} instead of chat/generate (or {@link LLM#embed} after
 * {@link LLM#builder} for the shared CPU pool). Speech models expose
 * {@link #transcribe} / {@link LLM#transcribe}. Synthesis models expose
 * {@link #synthesize} / {@link LLM#synthesize}.
 *
 * <p>{@link #modalities()} is what the checkpoint file declares (Gemma 4 QAT mobile includes
 * image, audio, and video keys). {@link #usableModalities()} is what this library actually
 * runs: text→text chat, text→embedding, audio→text Whisper, or text→audio Piper. Extra towers
 * are skipped at load.
 *
 * <p>GGUF models keep quantized weights packed by default. Prefer unpacking at load with
 * {@link LlmModelFactory#make(Path, LlmListener, boolean)} ({@code true}) so float32 is built
 * directly from the file with no packed heap copy. {@link LLM.Builder#allowUnpackParameters()}
 * late-unpacks an already-packed causal model by installing a dense graph. When no engine is
 * bound yet, packed payloads are released as each tensor is materialized. Packed tensors already
 * bound by existing engines are left intact (peak RAM may briefly hold packed + dense).
 *
 * <p>Construct via {@link LlmModelFactory#make(Path)}. Closing an {@link LLM} does not unload
 * this model — call {@link #close()} after every bound {@link LLM} is closed.
 *
 * <p>This type is {@code sealed}. The factory returns the library implementation; application
 * code cannot subclass it. The transformer graph stays in a non-exported type.
 *
 * @see LlmModelFactory
 * @see LLM
 */
public abstract sealed class LlmModel implements AutoCloseable permits LlmModelImpl {

  /**
   * Factory option: a {@link ThinkTags} pair for chat parse and ChatML skip-seed.
   *
   * @since 1.1.0
   */
  public static final String OPTION_THINK_TAGS = "thinkTags";

  /**
   * Factory option: {@link ChatSpecials} searched in decoded assistant text when stripping
   * leftover chat markup from the visible answer.
   *
   * @since 1.1.0
   */
  public static final String OPTION_CHAT_SPECIALS = "chatSpecials";

  /**
   * Factory option: a {@code Map<String, Object>} of {@link LlmOptionalData} values (for example
   * {@link LlmOptionalData#ESPEAK_DATA}). Prefer
   * {@link LlmModelFactory.Builder#optionalData(LlmOptionalData.Key, Object)}. Omitted when empty.
   *
   * @since 1.3.0
   */
  public static final String OPTION_OPTIONAL_DATA = "optionalData";

  protected LlmModel() {
  }

  /**
   * Checkpoint folder or GGUF file path this model was loaded from.
   */
  public abstract Path path();

  /**
   * Hugging Face / GGUF-mapped architecture config.
   */
  public abstract Config.HfConfig hfConfig();

  /**
   * Tokenizer bound to this checkpoint.
   */
  public abstract Tokenizer tokenizer();

  /**
   * Load-time options frozen by {@link LlmModelFactory} ({@link Map#copyOf}). Always contains
   * {@link #OPTION_THINK_TAGS} and {@link #OPTION_CHAT_SPECIALS} (library defaults when the caller
   * omitted them).
   *
   * @since 1.1.0
   */
  public abstract Map<String, Object> options();

  /**
   * Scratchpad open/close markers for chat parse and ChatML skip-seed.
   * {@link ThinkTags#DEFAULT} ({@code <think>} / {@code </think>}) unless
   * {@link #OPTION_THINK_TAGS} was set at load.
   *
   * @since 1.1.0
   */
  public abstract ThinkTags thinkTags();

  /**
   * Special-token strings searched in decoded assistant text when stripping chat markup.
   * {@link ChatSpecials#DEFAULT} unless {@link #OPTION_CHAT_SPECIALS} was set at load.
   *
   * @since 1.1.0
   */
  public abstract ChatSpecials chatSpecials();

  /**
   * Load-time extra for {@code key} (Piper espeak-ng-data folder, or an application key). Empty
   * when the caller did not set it. Safe to call after {@link #close()}.
   *
   * @param key typed extra; must not be {@code null}
   * @return the stored value, or empty
   * @throws NullPointerException if {@code key} is {@code null}
   * @since 1.3.0
   */
  public abstract <T> Optional<T> optionalData(LlmOptionalData.Key<T> key);

  /**
   * Architecture id (e.g. {@code qwen3}, {@code gemma3}, {@code bert}). Safe to call after
   * {@link #close()}.
   */
  public abstract String architectureName();

  /**
   * Content types declared by the checkpoint. Chat graphs always include text; Gemma 4 QAT mobile
   * also declares image, audio, and video input. Embedding encoders are
   * {@link LlmModalities#TEXT_TO_EMBEDDING}. Whisper speech models are
   * {@link LlmModalities#AUDIO_TO_TEXT}. Gemma 4 extra towers are skipped at load — see
   * {@link #usableModalities()}. Safe to call after {@link #close()}.
   *
   * @since 1.2.0
   */
  public abstract LlmModalities modalities();

  /**
   * Content types this library actually consumes and produces for the loaded graph: text→text
   * chat, text→embedding, audio→text speech, or text→audio synthesis. Safe to call after
   * {@link #close()}.
   *
   * @since 1.2.0
   */
  public abstract LlmModalities usableModalities();

  /**
   * Modalities this checkpoint consumes.
   *
   * @since 1.2.0
   */
  public final Set<LlmModality> inputModalities() {
    return this.modalities().input();
  }

  /**
   * Modalities this checkpoint produces.
   *
   * @since 1.2.0
   */
  public final Set<LlmModality> outputModalities() {
    return this.modalities().output();
  }

  /**
   * {@code true} when this file is for chat / text completion — use {@link LLM#builder(LlmModel)}.
   *
   * @since 1.1.0
   */
  public abstract boolean isCausalModel();

  /**
   * {@code true} when this file turns text into number vectors — use {@link LLM#builder(LlmModel)}
   * then {@link LLM#embed}. {@link #embed(CharSequence)} remains a sequential shortcut.
   *
   * @since 1.1.0
   */
  public abstract boolean isEmbeddingModel();

  /**
   * {@code true} when this file is Whisper speech-to-text — use {@link LLM#builder(LlmModel)}
   * then {@link LLM#transcribe}. {@link #transcribe} remains a sequential shortcut.
   *
   * @since 1.3.0
   */
  public abstract boolean isSpeechModel();

  /**
   * {@code true} when this file is Piper (or other) text-to-speech — use {@link LLM#builder(LlmModel)}
   * then {@link LLM#synthesize}. {@link #synthesize} remains a sequential shortcut.
   *
   * @since 1.3.0
   */
  public abstract boolean isSynthesisModel();

  /**
   * {@code true} when GGUF/QAT weights are still packed (not widened to float32).
   */
  public abstract boolean hasPackedWeights();

  /**
   * {@code true} after {@link #close()} (further accessors throw).
   */
  public abstract boolean isClosed();

  /**
   * Kind, architecture, modalities, container, sizes, packed/dense/qat, and chat format
   * (safe after close).
   *
   * @since 1.1.0
   */
  @Override
  public abstract String toString();

  /**
   * Turns {@code text} into a number vector for similarity / search (embedding models only).
   * The vector is L2-normalized. Tokenizes and wraps with {@code [CLS]} / {@code [SEP]}
   * (or XLM-R {@code <s>} / {@code </s>}) when those tokens exist. Safe to call concurrently.
   * Some embedding checkpoints expect a prefix such as {@code "query: "} — that is a model
   * convention, not inserted by this method.
   *
   * @since 1.1.0
   */
  public abstract float[] embed(final CharSequence text);

  /**
   * Encodes each text to an L2-normalized embedding vector (embedding models only).
   * Safe to call concurrently.
   *
   * @since 1.1.0
   */
  public abstract float[][] embed(final List<? extends CharSequence> texts);

  /**
   * Encodes already-tokenized ids to a single L2-normalized embedding (embedding models only).
   * Ids are used as-is — include special tokens such as {@code [CLS]} / {@code [SEP]}
   * (or {@code <s>} / {@code </s>}) when required.
   *
   * @since 1.1.0
   */
  public abstract float[] embed(final int[] tokenIds);

  /**
   * Transcribes an uncompressed WAV file (PCM or IEEE float, mixed to mono). Speech models only.
   * Prefer {@link #transcribe(byte[])} when the payload is already in memory. Sequential matmul
   * unless you go through {@link LLM#transcribe}.
   *
   * @param wav path to a {@code .wav} file; must not be {@code null}
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code wav} is {@code null}
   * @throws IOException              if the file cannot be read
   * @throws IllegalArgumentException if the container is compressed or malformed
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final Path wav) throws IOException;

  /**
   * Transcribes an uncompressed WAV file with an optional language hint. {@code null} or
   * {@link Locale#ROOT} selects automatically; region is ignored ({@link Locale#US} is English).
   * Speech models only.
   *
   * @param wav      path to a {@code .wav} file; must not be {@code null}
   * @param language hint, or {@code null}/{@link Locale#ROOT} for auto
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code wav} is {@code null}
   * @throws IOException              if the file cannot be read
   * @throws IllegalArgumentException if the container is compressed or malformed, or the language
   *                                  is not a Whisper token
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final Path wav, final Locale language) throws IOException;

  /**
   * Transcribes an uncompressed WAV payload already in memory (same container as a {@code .wav}
   * file). Speech models only. Does not touch the filesystem.
   *
   * @param wav RIFF/WAVE bytes; must not be {@code null}
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code wav} is {@code null}
   * @throws IllegalArgumentException if the container is compressed or malformed
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final byte[] wav);

  /**
   * Transcribes in-memory WAV bytes with an optional language hint. Speech models only.
   *
   * @param wav      RIFF/WAVE bytes; must not be {@code null}
   * @param language hint, or {@code null}/{@link Locale#ROOT} for auto
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code wav} is {@code null}
   * @throws IllegalArgumentException if the container is compressed or malformed, or the language
   *                                  is not a Whisper token
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final byte[] wav, final Locale language);

  /**
   * Transcribes mono PCM. {@code sampleRate} is Hertz; other rates are resampled to 16 kHz.
   * Speech models only.
   *
   * @param pcm        mono samples in {@code [-1, 1]}; must not be {@code null}
   * @param sampleRate Hertz of {@code pcm}; must be {@code >= 1}
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code pcm} is {@code null}
   * @throws IllegalArgumentException if {@code sampleRate < 1}
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final float[] pcm, final int sampleRate);

  /**
   * Transcribes mono PCM with an optional language hint. Speech models only.
   *
   * @param pcm        mono samples in {@code [-1, 1]}; must not be {@code null}
   * @param sampleRate Hertz of {@code pcm}; must be {@code >= 1}
   * @param language   hint, or {@code null}/{@link Locale#ROOT} for auto
   * @return transcript text; never {@code null}
   * @throws NullPointerException     if {@code pcm} is {@code null}
   * @throws IllegalArgumentException if {@code sampleRate < 1}, or the language is not a Whisper
   *                                  token
   * @throws IllegalStateException    if this model is closed or is not speech
   * @since 1.3.0
   */
  public abstract String transcribe(final float[] pcm, final int sampleRate, final Locale language);

  /**
   * Synthesizes uncompressed WAV bytes (PCM16 little-endian, mono) from {@code text}.
   * The payload is the file contents — write it only if you need a {@code .wav} on disk.
   * Synthesis models only. Sequential matmul unless you go through {@link LLM#synthesize}.
   *
   * @param text text to speak; must not be {@code null} or blank
   * @return RIFF/WAVE bytes; never {@code null}
   * @throws NullPointerException     if {@code text} is {@code null}
   * @throws IllegalArgumentException if {@code text} is blank
   * @throws IllegalStateException    if this model is closed or is not synthesis
   * @since 1.3.0
   */
  public abstract byte[] synthesize(final CharSequence text);

  /**
   * Releases packed payloads, closes weight file channels (&gt; 2 GiB shards), and drops
   * weight/network/encoder refs so the heap can reclaim them. Close every bound {@link LLM}
   * first; closing the model while an engine still holds the graph leaves dense arrays reachable
   * until that engine closes. Idempotent. Dense {@code float[]} buffers are not zeroed — they
   * become GC-only after this call.
   */
  @Override
  public abstract void close();
}
