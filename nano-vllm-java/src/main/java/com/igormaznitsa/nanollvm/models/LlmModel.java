package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.LlmModelImpl;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Loaded model: architecture weights, HF/GGUF config, and tokenizer.
 *
 * <p>Safe to share across threads and across many {@link LLM} instances (causal models). Mutable
 * inference state (KV cache, scheduler, sampling) lives on each {@link LLM}, not here. Load-time
 * options ({@link #options()}, including {@link #OPTION_THINK_TAGS} and
 * {@link #OPTION_CHAT_SPECIALS}) are frozen at
 * {@link LlmModelFactory#make} and never change. Embedding models expose
 * {@link #embed(CharSequence)} instead of chat/generate.
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
   * Architecture id (e.g. {@code qwen3}, {@code gemma3}, {@code bert}). Safe to call after
   * {@link #close()}.
   */
  public abstract String architectureName();

  /**
   * {@code true} when this file is for chat / text completion — use {@link LLM#builder(LlmModel)}.
   *
   * @since 1.1.0
   */
  public abstract boolean isCausalModel();

  /**
   * {@code true} when this file turns text into number vectors — call {@link #embed(CharSequence)},
   * not {@link LLM#builder(LlmModel)} (that {@code build()} rejects embedding models).
   *
   * @since 1.1.0
   */
  public abstract boolean isEmbeddingModel();

  /**
   * {@code true} when GGUF/QAT weights are still packed (not widened to float32).
   */
  public abstract boolean hasPackedWeights();

  /**
   * {@code true} after {@link #close()} (further accessors throw).
   */
  public abstract boolean isClosed();

  /**
   * Kind, architecture, container, sizes, packed/dense/qat, and chat format (safe after close).
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
   * Releases packed payloads, closes weight file channels (&gt; 2 GiB shards), and drops
   * weight/network/encoder refs so the heap can reclaim them. Close every bound {@link LLM}
   * first; closing the model while an engine still holds the graph leaves dense arrays reachable
   * until that engine closes. Idempotent. Dense {@code float[]} buffers are not zeroed — they
   * become GC-only after this call.
   */
  @Override
  public abstract void close();
}
