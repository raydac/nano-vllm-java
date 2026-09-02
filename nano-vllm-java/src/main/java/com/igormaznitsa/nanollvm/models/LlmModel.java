package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.LlmModelImpl;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
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
 * {@link LlmModelFactory#make} and never change. Embedding, speech, synthesis, and classification
 * graphs use
 * {@link #generate(LlmInput, LlmModality)} / {@link LLM#generate(LlmInput, LlmModality)} for typed
 * {@link LlmOutput} results (text, sound, embedding, or labels). Chat dialog stays on
 * {@link LLM#chat} / {@link com.igormaznitsa.nanollvm.chat.ChatSession}; batched token generation
 * stays on {@link LLM#generate(java.util.List, com.igormaznitsa.nanollvm.llm.SamplingParams)}.
 *
 * <p>{@link #modalities()} is what the checkpoint file declares (Gemma 4 QAT mobile includes
 * image, audio, and video keys). {@link #usableModalities()} is what this library actually
 * runs: text→text chat, text→embedding, audio→text Whisper, text→audio Piper, or text→labels
 * fastText. Extra towers are skipped at load.
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
   * {@link LlmModalities#AUDIO_TO_TEXT}. Piper synthesis models are
   * {@link LlmModalities#TEXT_TO_AUDIO}. fastText classifiers are
   * {@link LlmModalities#TEXT_TO_LABELS}. Gemma 4 extra towers are skipped at load — see
   * {@link #usableModalities()}. Safe to call after {@link #close()}.
   *
   * @since 1.2.0
   */
  public abstract LlmModalities modalities();

  /**
   * Content types this library actually consumes and produces for the loaded graph: text→text
   * chat, text→embedding, audio→text speech, text→audio synthesis, or text→labels
   * classification. Safe to call after {@link #close()}.
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
   * then {@link LLM#generate(LlmInput, LlmModality)} with {@link LlmModality#EMBEDDING}.
   * {@link #generate(LlmInput, LlmModality)} remains a sequential shortcut.
   *
   * @since 1.1.0
   */
  public abstract boolean isEmbeddingModel();

  /**
   * {@code true} when this file is Whisper speech-to-text — use {@link LLM#builder(LlmModel)}
   * then {@link LLM#generate(LlmInput, LlmModality)} with {@link LlmInSound} → {@link LlmModality#TEXT}.
   * {@link #generate(LlmInput, LlmModality)} remains a sequential shortcut.
   *
   * @since 1.3.0
   */
  public abstract boolean isSpeechModel();

  /**
   * {@code true} when this file is Piper (or other) text-to-speech — use {@link LLM#builder(LlmModel)}
   * then {@link LLM#generate(LlmInput, LlmModality)} with {@link LlmInText} → {@link LlmModality#AUDIO}.
   * {@link #generate(LlmInput, LlmModality)} remains a sequential shortcut.
   *
   * @since 1.3.0
   */
  public abstract boolean isSynthesisModel();

  /**
   * {@code true} when this file is fastText (or other) text classification — use
   * {@link LLM#builder(LlmModel)} then {@link LLM#generate(LlmInput, LlmModality)} with
   * {@link LlmInText} → {@link LlmModality#LABELS} ({@link LlmOutLabels}).
   * {@link #generate(LlmInput, LlmModality)} remains a sequential shortcut.
   *
   * @since 1.4.0
   */
  public abstract boolean isClassificationModel();

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
   * Cross-kind entry: runs the graph that produces {@code outputModality} from {@code input}.
   * Sequential matmul unless you go through {@link LLM#generate(LlmInput, LlmModality)}.
   *
   * <p>Supported pairs on this model shortcut:
   * <ul>
   *   <li>{@link LlmInText} → {@link LlmModality#EMBEDDING} — embedding encoders</li>
   *   <li>{@link LlmInTokenIds} → {@link LlmModality#EMBEDDING} — already-tokenized ids</li>
   *   <li>{@link LlmInText} → {@link LlmModality#AUDIO} — Piper synthesis ({@link LlmOutSoundData})</li>
   *   <li>{@link LlmInSound} → {@link LlmModality#TEXT} — Whisper transcription</li>
   *   <li>{@link LlmInText} → {@link LlmModality#LABELS} — fastText classification ({@link LlmOutLabels})</li>
   * </ul>
   * Text completion ({@link LlmInText} → {@link LlmModality#TEXT}) needs an {@link LLM} engine —
   * call {@link LLM#generate(LlmInput, LlmModality)} instead.
   *
   * @param input          typed payload; must not be {@code null}
   * @param outputModality desired result type; must not be {@code null}
   * @return typed result matching {@code outputModality}
   * @throws NullPointerException     if either argument is {@code null}
   * @throws IllegalArgumentException if the input/output pair is unsupported
   * @throws IllegalStateException    if this model is closed or the wrong graph kind
   * @since 1.3.0
   */
  public abstract LlmOutput generate(final LlmInput input, final LlmModality outputModality);

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
