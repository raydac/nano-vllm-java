package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.engine.Scheduler;
import com.igormaznitsa.nanollvm.engine.Sequence;
import com.igormaznitsa.nanollvm.engine.Transformer;
import com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException;
import com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.LlmModelImpl;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * One engine bound to a loaded chat/completion {@link LlmModel}: conversation, raw continuation,
 * batch generate, and RAG. Embedding files use {@link LlmModel#embed}, not this type.
 *
 * <h2>Typical use</h2>
 * <pre>{@code
 * try (LlmModel model = LlmModelFactory.open(modelDir).make();  // load once, share freely
 *      LLM llm = LLM.builder(model)
 *         .maxModelLen(2048)
 *         .sampling(SamplingParams.builder().temperature(0.2f).maxTokens(128).build())
 *         .systemPrompt("Answer briefly.")  // optional
 *         .advisors(LlmAdvisorMixer.defaults(),
 *             LlmAdvisor.builder().name("Facts").prompt("List key facts from the request.").build(),
 *             LlmAdvisor.builder().name("Risks").prompt("Note risks or contradictions.").build())
 *         .build()) {
 *     String reply = llm.chat().send("Hello").answer();
 *     String once = llm.chatOnce("What is 2+2?", 64);
 *     String raw = llm.complete("The capital of France is", 32);
 * }  // LLM closes first, then model
 * }</pre>
 *
 * <h2>If you want…</h2>
 * A <em>token</em> is a word-piece the model counts (often a bit less than a word). Reply length
 * and memory limits are in tokens, not characters.
 * <ul>
 *   <li><b>A conversation that remembers earlier turns</b> — {@link #chat()} then
 *       {@link ChatSession#send(String)}; show {@link ChatReply#answer()} to the user.</li>
 *   <li><b>One question, no history</b> — {@link #chatOnce(String)}.</li>
 *   <li><b>Finish a sentence / continue a text as-is</b> — {@link #complete(String)}
 *       (no chat role wrapping).</li>
 *   <li><b>Answer from your files</b> — index with {@link RagFactory}, then {@link #rag(RagIndex)}.</li>
 *   <li><b>Shorter or longer replies</b> — {@link SamplingParams.Builder#maxTokens(int)} via
 *       {@link Builder#sampling(SamplingParams)}, or {@link #chat(int)} /
 *       {@link ChatSession#maxTokens(int)}.</li>
 *   <li><b>More predictable wording</b> — lower {@link SamplingParams.Builder#temperature(float)}
 *       ({@code 0.1}–{@code 0.2} for facts / RAG; default {@code 0.6}). {@code 0} is rejected.</li>
 *   <li><b>Faster CPU</b> — {@link Builder#cpuThreads(int)} / {@link Builder#allCpuThreads()};
 *       for packed GGUF also {@link LlmModelFactory.Builder#unpackParameters()} at load
 *       (more RAM, faster math).</li>
 *   <li><b>Less RAM</b> — smaller {@link Builder#maxModelLen(int)} and
 *       {@link Builder#kvHeapFraction(float)}; keep GGUF packed (default).</li>
 *   <li><b>Stop a stuck generate</b> — {@link #cancel()} from another thread.</li>
 *   <li><b>A number vector for search / similarity</b> — {@link LlmModel#embed} on an embedding
 *       checkpoint; {@link Builder#build()} rejects those files.</li>
 * </ul>
 * Prefer {@link #chat()} / {@link #complete} / {@link #chatOnce} unless you need a batch of prompts
 * or a token-id stream — that is {@link #generate} / {@link #generateTokenIds}.
 *
 * <h2>Defaults</h2>
 * Construction is <em>library-quiet</em> ({@link LlmListeners#silent()}). CLI tools should pass
 * {@link LlmListeners#toSystem()} via {@link Builder#listen(LlmListener)}. Architecture is auto-detected from
 * {@code config.json}
 * (override with {@code -Dnanollvm.arch=qwen3|gemma3|llama|lfm2}).
 *
 * <h2>Thread safety</h2>
 * <ul>
 *   <li>One {@code LLM} must not run concurrent {@link #generate}, chat, RAG, or {@link #complete}
 *       — those share one generate lock / scheduler.</li>
 *   <li>Prefer one instance per thread, or external locking. Share one immutable {@link LlmModel}
 *       across many {@code LLM}s.</li>
 *   <li>{@link #cancel()} is safe from another thread and aborts an in-flight generate with
 *       {@link GenerationCancelledException}.</li>
 *   <li>Do not call {@link #generate}, {@link #runAdvisors}, or chat/RAG from an {@code onToken}
 *       callback — that re-enters the generate lock and deadlocks.</li>
 *   <li>After {@link #close()}, do not call generate/chat APIs on this instance. Closing this
 *       engine does not unload {@link #model()} — close the {@link LlmModel} separately when done.</li>
 * </ul>
 *
 * <h2>Servers, threads, and memory</h2>
 * One {@code LLM} is a single in-flight generate (chat / RAG / complete share that lock). For
 * concurrent HTTP requests, share one {@link LlmModel} and keep a pool of {@code LLM} engines
 * (or serialize onto one engine). {@link #cancel()} is the cross-thread abort; cap wall time with
 * {@link #generate(List, SamplingParams, Duration)} or {@link ChatSession#timeout(Duration)}.
 *
 * <p>Default {@code cpuThreads > 1} joins a <em>process-wide</em> daemon matmul pool named
 * {@code nanollvm-matmul-*} (sized to {@link Runtime#availableProcessors()}, not to your
 * {@code cpuThreads} cap). That pool also runs independent attention, RoPE, and embedding-gather
 * chunks. It is unexpected in a cloud JVM that already owns executors.
 * Pin work:
 * <ul>
 *   <li>{@link Builder#matmulExecutor(ExecutorService)} — your pool; this engine does not shut it down</li>
 *   <li>{@link Builder#dedicatedMatmulPool()} — a bounded pool of {@code cpuThreads} workers this
 *       engine {@linkplain #close() closes}</li>
 *   <li>{@link Builder#disableMultiCpu()} — calling thread only; no matmul executor</li>
 * </ul>
 * Dense RAG indexing is sequential unless you pass an {@link java.util.concurrent.Executor};
 * {@link LlmModel#embed} runs on the caller. Load is blocking I/O on the calling thread — do it
 * at startup. Close engines, then the model. Files &gt; 2 GiB keep a {@code FileChannel} until
 * model close; dense weight arrays become unreachable for GC after close (they are not zeroed).
 * {@link com.igormaznitsa.nanollvm.utils.ResourceLimits#setCurrent} is JVM-global; per-corpus
 * caps belong on {@link com.igormaznitsa.nanollvm.rag.RagLoadOptions}.
 *
 * <h2>Prompt and sampling restrictions</h2>
 * <ul>
 *   <li>Prompt lists must be non-{@code null}; each entry non-{@code null}. Empty list returns
 *       an empty result list.</li>
 *   <li>When passing per-prompt {@link SamplingParams}, list sizes must match.</li>
 *   <li>{@link SamplingParams}: {@code temperature > 1e-10}, {@code maxTokens >= 1},
 *       {@code topK >= 0} ({@code 0} = off), {@code topP in (0, 1]}.</li>
 *   <li>Prompt length + {@code maxTokens} should fit {@link Builder#maxModelLen(int)} (and the
 *       model's position limit); oversized contexts fail or truncate at the scheduler / KV layer.</li>
 *   <li>Batch size is limited by {@link Builder#maxNumSeqs(int)} and prefill by
 *       {@link Builder#maxNumBatchedTokens(int)}.</li>
 * </ul>
 *
 * @see Builder
 * @see LlmModel
 * @see LlmModelFactory
 * @see ChatSession
 * @see LlmListener
 * @see SamplingParams
 * @see RagFactory
 * @see PreparedRag
 */
public final class LLM implements AutoCloseable {

  private static final int WARMUP_PREFILL_TOKENS = 64;
  private static final int WARMUP_DECODE_TOKENS = 16;

  private final LlmModel model;
  private final Config config;
  private final LlmListener listener;
  private final String systemPromptOverride;
  private final MatmulRuntime matmul;
  private final Transformer transformer;
  private final Tokenizer tokenizer;
  private final Scheduler scheduler;
  private final ReentrantLock generateLock = new ReentrantLock();
  private final AtomicBoolean cancelRequested = new AtomicBoolean();
  private final AtomicBoolean closed = new AtomicBoolean();
  private final List<LlmAdvisor> advisors;
  private final LlmAdvisorMixer advisorMixer;
  private final Predicate<String> advisorNoteFilter;
  private final SamplingParams defaultSampling;

  private LLM(final Builder builder) {
    requireNonNull(builder, "builder");
    MatmulRuntime createdMatmul = null;
    Transformer createdTransformer = null;
    LlmModel boundModel = null;
    boolean acquired = false;
    try {
      int cpuThreads = builder.resolveCpuThreads();
      MatmulRuntime.Builder matmulBuilder = MatmulRuntime.builder().cpuThreads(cpuThreads);
      if (cpuThreads > 1 && builder.matmulExecutor != null) {
        matmulBuilder.executor(builder.matmulExecutor);
      } else if (cpuThreads > 1 && builder.dedicatedMatmulPool) {
        matmulBuilder.dedicatedPool();
      }
      createdMatmul = matmulBuilder.build();
      this.matmul = createdMatmul;
      boundModel = builder.resolveModel();
      this.model = boundModel;
      this.tokenizer = this.model.tokenizer();
      this.config = builder.toConfig(this.model, this.tokenizer, cpuThreads);
      this.listener = builder.listener;
      this.systemPromptOverride = builder.systemPromptOverride;
      this.advisors = builder.advisors;
      this.advisorMixer = builder.advisorMixer;
      this.advisorNoteFilter = builder.advisorNoteFilter;
      this.defaultSampling = builder.samplingParams == null
        ? SamplingDefaults.neutral()
        : builder.samplingParams;
      createdTransformer = new Transformer(
        this.model, this.config, this.matmul, this.listener, builder.allowUnpackParameters);
      this.transformer = createdTransformer;
      this.scheduler = new Scheduler(this.config, this.transformer::clearConvState);
      if (builder.warmup) {
        this.warmup();
      }
      LlmModelImpl.peer(this.model).acquireEngine();
      acquired = true;
      LlmListeners.info(this.listener, this, "CPU matmul: " + this.matmul.backendInfo());
      LlmListeners.info(this.listener, this, this.model.hasPackedWeights()
        ? "Weights: GGUF packed (specialized LinearKernel / EmbeddingKernel per tensor)"
        : "Weights: float32 dense (decode-1 + parallel MatmulRuntime)");
    } catch (ModelLoadException e) {
      abortConstruction(createdMatmul, createdTransformer, boundModel, acquired);
      throw e;
    } catch (RuntimeException e) {
      abortConstruction(createdMatmul, createdTransformer, boundModel, acquired);
      throw new ModelLoadException("failed to load model from " + builder.modelPath(), e);
    }
  }

  private static void abortConstruction(
    final MatmulRuntime createdMatmul,
    final Transformer createdTransformer,
    final LlmModel boundModel,
    final boolean acquired
  ) {
    try {
      if (createdTransformer != null) {
        createdTransformer.close();
      }
    } finally {
      if (createdMatmul != null) {
        createdMatmul.close();
      }
      if (acquired && boundModel != null) {
        LlmModelImpl.peer(boundModel).releaseEngine();
      }
    }
  }

  /**
   * Starts a fluent configurator for a shared immutable {@link LlmModel}. This engine does not
   * close the model — close {@link LLM} first, then the model. Embedding checkpoints are rejected
   * at {@link Builder#build()}; use {@link LlmModel#embed} instead.
   *
   * @param model loaded model to bind; not closed by this {@code LLM}; must be non-{@code null}
   * @return a new builder; call {@link Builder#build()} to construct the engine
   * @throws NullPointerException if {@code model} is {@code null}
   */
  public static Builder builder(final LlmModel model) {
    return new Builder(requireNonNull(model, "model"));
  }

  /**
   * The immutable loaded model bound to this engine. Safe to share with other {@code LLM}s; this
   * instance does not own it. Closing this engine does not unload weights.
   *
   * @return the model passed to the builder; never {@code null}
   */
  public LlmModel model() {
    return this.model;
  }

  private void warmup() {
    // JIT / cache warm-up: one short generate so the first real request is not cold
    LlmListeners.info(this.listener, this, "Warming up (prefill + decode)…");
    long startedAtNanos = System.nanoTime();
    this.generateTokenIds(
      List.of(this.syntheticWarmupPrompt()),
      SamplingParams.builder()
        .temperature(0.6f)
        .maxTokens(WARMUP_DECODE_TOKENS)
        .ignoreEos(true)
        .build(),
      false,
      Duration.ZERO,
      null);
    LlmListeners.infof(this.listener, this, "Warmup done in %.1fs%n",
      (System.nanoTime() - startedAtNanos) / 1e9);
  }

  private List<Integer> syntheticWarmupPrompt() {
    // Fixed synthetic token ids — shape matters for warmup, not linguistic content
    return IntStream.range(0, WARMUP_PREFILL_TOKENS)
      .map(i -> 1 + (i % 97))
      .boxed()
      .toList();
  }

  private StepResult stepUnlocked() {
    // Business tick: pick a prefill or decode batch, run the model, commit tokens

    // 1) Scheduler chooses which sequences run this tick (prefill vs decode)
    Scheduler.ScheduleResult scheduled = this.scheduler.schedule();

    // 2) Forward pass + sampling → one next-token id per scheduled sequence
    List<Integer> nextTokenIds = this.runForwardAndSample(scheduled);

    // 3) Append tokens, update KV bookkeeping, finish sequences that hit stop / maxTokens
    List<int[]> appendedTokens = this.applySchedulerPostprocess(scheduled, nextTokenIds);

    // 4) Package finished completions, per-token events, and workload metric for callers
    return new StepResult(
      this.collectFinishedOutputs(scheduled.sequences()),
      this.toTokenEvents(appendedTokens),
      this.measureStepWorkload(scheduled));
  }

  private List<Integer> runForwardAndSample(final Scheduler.ScheduleResult scheduled) {
    // Internal: Transformer.step → CausalLM forward → logits → Sampler
    return this.transformer.step(scheduled.sequences(), scheduled.prefill());
  }

  private List<int[]> applySchedulerPostprocess(
    final Scheduler.ScheduleResult scheduled,
    final List<Integer> nextTokenIds
  ) {
    // Internal: write sampled ids into sequences; collect (seqId, tokenId) for streaming
    List<int[]> appendedTokens = new ArrayList<>();
    this.scheduler.postprocess(
      scheduled.sequences(), nextTokenIds, scheduled.prefill(), appendedTokens);
    return appendedTokens;
  }

  private List<FinishedOutput> collectFinishedOutputs(final List<Sequence> sequences) {
    // Internal: only sequences that reached stop / maxTokens on this tick
    return sequences.stream()
      .filter(Sequence::isFinished)
      .map(seq -> new FinishedOutput(
        seq.seqId(), seq.completionTokenIds(), seq.numPromptTokens()))
      .toList();
  }

  private List<TokenEvent> toTokenEvents(final List<int[]> appendedTokens) {
    // Internal: raw [seqId, tokenId] pairs → typed stream events
    return appendedTokens.stream()
      .map(pair -> new TokenEvent(pair[0], pair[1]))
      .toList();
  }

  private int measureStepWorkload(final Scheduler.ScheduleResult scheduled) {
    // Internal: +token count for prefill; −batch size for decode (progress convention)
    return scheduled.prefill()
      ? scheduled.sequences().stream().mapToInt(Sequence::numScheduledTokens).sum()
      : -scheduled.sequences().size();
  }

  /**
   * Requests cancellation of an in-flight {@link #generate} (and chat/RAG/complete that use it).
   * Safe to call from other threads. The blocked generate then throws
   * {@link GenerationCancelledException}.
   *
   * @apiNote A cancel posted <em>before</em> {@code generate} begins is cleared when that generate
   * starts; only in-flight work is aborted. After cancel, leftover KV pages are released when
   * generate unwinds.
   */
  public void cancel() {
    this.cancelRequested.set(true);
  }

  private static Consumer<TokenEvent> adaptTokenCallback(final IntConsumer onToken) {
    if (onToken == null) {
      return null;
    }
    return event -> onToken.accept(event.tokenId());
  }

  /**
   * Completes each prompt string with shared sampling: no progress line, no timeout, no token
   * stream. Use this when you only need finished text. Strings are <em>not</em> wrapped in a chat
   * template — for dialog use {@link #chat()} / {@link #chatOnce(String)}.
   *
   * <p>Exclusive on this instance (see class-level thread-safety). Empty {@code prompts} returns
   * an empty list. {@code maxTokens} is clamped to remaining {@link Config#maxModelLen()}.
   *
   * @param prompts        one string per sequence; non-{@code null}; elements non-{@code null};
   *                       size limited by {@link Config#maxNumSeqs()}
   * @param samplingParams shared sampling for every prompt; non-{@code null}
   * @return one {@link GenerationOutput} per prompt, in completion order by sequence id
   * @throws GenerationCancelledException if {@link #cancel()} fires during the run
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   * @throws IllegalStateException        if this engine is {@linkplain #isClosed() closed}
   * @see #generate(List, SamplingParams, boolean, Duration, IntConsumer)
   */
  public List<GenerationOutput> generate(final List<String> prompts,
                                         final SamplingParams samplingParams) {
    return this.generate(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * Same as {@link #generate(List, SamplingParams)} with a wall-clock limit. {@code null}, zero,
   * or negative {@code timeout} means no limit. On expiry the batch throws
   * {@link GenerationTimeoutException} and leftover KV pages are released.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param timeout        max wall time for the whole batch; {@code null} / zero / negative = none
   * @return one output per prompt
   * @throws GenerationTimeoutException   if {@code timeout} elapses before the batch finishes
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @since 1.1.0
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final Duration timeout
  ) {
    return this.generate(prompts, samplingParams, false, timeout, null);
  }

  /**
   * Same as {@link #generate(List, SamplingParams)} with a per-token <em>id</em> callback and no
   * timeout. {@code onToken} sees every new token across the batch but not which prompt it belongs
   * to — use {@link #generate(List, SamplingParams, Consumer)} for {@link TokenEvent#seqId()}.
   * Must not call generate / chat / {@link #runAdvisors} from the callback (deadlock).
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param onToken        invoked for each newly decoded token id ({@code null} = no stream)
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final IntConsumer onToken
  ) {
    return this.generate(prompts, samplingParams, false, Duration.ZERO, onToken);
  }

  /**
   * Same as {@link #generate(List, SamplingParams)} with a seq-aware token callback and no timeout.
   * {@link TokenEvent#seqId()} matches the prompt index in {@code prompts}. Must not re-enter
   * generate from the callback.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param onToken        per-token events ({@code null} = no stream)
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final Consumer<TokenEvent> onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(prompts, samplingParams, false, Duration.ZERO, onToken);
  }

  /**
   * Text-prompt generate with a wall-clock limit and a seq-aware token callback. Combines
   * {@link #generate(List, SamplingParams, Duration)} and
   * {@link #generate(List, SamplingParams, Consumer)}.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param timeout        max wall time; {@code null} / zero / negative = none
   * @param onToken        per-token events ({@code null} = no stream); must not re-enter generate
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final Duration timeout,
    final Consumer<TokenEvent> onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(prompts, samplingParams, false, timeout, onToken);
  }

  /**
   * Completes each prompt string with shared sampling and an optional in-place progress line.
   * {@code useTqdm} means “show a progress bar” (the name is historical). Progress is emitted only
   * when that flag is true <em>and</em> {@link #listener()} is not silent — the library-default
   * silent listener prints nothing. For a timeout or a token stream without this flag, use
   * {@link #generate(List, SamplingParams, Duration)} or
   * {@link #generate(List, SamplingParams, Consumer)}.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param useTqdm        when {@code true} and the listener is not silent, emits progress via
   *                       {@link LlmListener}; ignored when {@link LlmListeners#isSilent(LlmListener)}
   * @return one output per prompt
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   */
  public List<GenerationOutput> generate(final List<String> prompts,
                                         final SamplingParams samplingParams,
                                         final boolean useTqdm) {
    return this.generate(prompts, samplingParams, useTqdm, Duration.ZERO, null);
  }

  /**
   * Completes each prompt with <em>its own</em> {@link SamplingParams} (temperature / max tokens
   * may differ). List sizes must match. No progress, timeout, or token stream — see the five-arg
   * overload for those knobs.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams one params object per prompt; sizes must match; non-{@code null}
   * @return one output per prompt
   * @throws IllegalArgumentException if list sizes differ
   * @since 1.1.0
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final List<SamplingParams> samplingParams
  ) {
    return this.generate(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * Generates completions for text prompts with an optional per-token callback and no timeout.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param useTqdm        progress line when listener is not silent
   * @param onToken        invoked for each newly decoded token id across the batch
   *                       ({@code null} = no streaming); must not call generate/chat/advisors
   * @return one output per prompt
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   * @apiNote Calling {@link #runAdvisors} or another {@code generate} from {@code onToken} deadlocks.
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final boolean useTqdm,
    final IntConsumer onToken
  ) {
    return this.generate(prompts, samplingParams, useTqdm, Duration.ZERO, onToken);
  }

  /**
   * Full text-prompt generate: enqueue → drive ticks until idle → decode completions.
   *
   * @param prompts        one string prompt per sequence; non-{@code null}; empty list → empty result
   * @param samplingParams shared sampling settings for all prompts; non-{@code null}
   * @param useTqdm        progress line when the listener is not silent
   * @param timeout        max wall time for the whole batch; {@code null}, {@link Duration#ZERO},
   *                       or negative means no limit
   * @param onToken        per-token callback across the batch, or {@code null}; must not re-enter generate
   * @return one {@link GenerationOutput} per input prompt (seq-id order)
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws GenerationTimeoutException   if {@code timeout} elapses before the batch finishes
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   * @apiNote Exclusive on this instance. Do not call {@link #runAdvisors} or another {@code generate}
   * from {@code onToken}.
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final SamplingParams samplingParams,
    final boolean useTqdm,
    final Duration timeout,
    final IntConsumer onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(
      prompts, samplingParams, useTqdm, timeout, LLM.adaptTokenCallback(onToken));
  }

  /**
   * Text-prompt generate with per-prompt sampling parameters.
   *
   * @param prompts        one string per sequence; non-{@code null}
   * @param samplingParams one {@link SamplingParams} per prompt; size must equal {@code prompts.size()};
   *                       non-{@code null}
   * @param useTqdm        progress when listener is not silent
   * @param timeout        wall-clock limit; {@code null}/zero/negative = none
   * @param onToken        optional streaming callback; must not re-enter generate
   * @return one output per prompt
   * @throws IllegalArgumentException     if list sizes differ
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws GenerationTimeoutException   if {@code timeout} elapses
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   */
  public List<GenerationOutput> generate(
    final List<String> prompts,
    final List<SamplingParams> samplingParams,
    final boolean useTqdm,
    final Duration timeout,
    final IntConsumer onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(
      prompts, samplingParams, useTqdm, timeout, LLM.adaptTokenCallback(onToken));
  }

  /**
   * Completes pre-tokenized prompts with shared sampling: no progress, timeout, or stream.
   * Each inner list is vocabulary ids from {@link #tokenizer()} (or an equivalent encode).
   * Prefer this when the caller already tokenized; {@link #generate(List, SamplingParams)}
   * encodes strings for you.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}; each list non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @return one output per prompt
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   */
  public List<GenerationOutput> generateTokenIds(final List<List<Integer>> prompts,
                                                 final SamplingParams samplingParams) {
    return this.generateTokenIds(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * Same as {@link #generateTokenIds(List, SamplingParams)} with a wall-clock limit.
   * {@code null} / zero / negative {@code timeout} means no limit.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param timeout        max wall time; {@code null} / zero / negative = none
   * @return one output per prompt
   * @throws GenerationTimeoutException if {@code timeout} elapses
   * @since 1.1.0
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final SamplingParams samplingParams,
    final Duration timeout
  ) {
    return this.generateTokenIds(prompts, samplingParams, false, timeout, null);
  }

  /**
   * Same as {@link #generateTokenIds(List, SamplingParams)} with a per-token id callback and no
   * timeout. The callback does not include {@code seqId} — use
   * {@link #generateTokenIds(List, SamplingParams, Consumer)} for seq-aware events.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param onToken        per-token ids ({@code null} = no stream); must not re-enter generate
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final SamplingParams samplingParams,
    final IntConsumer onToken
  ) {
    return this.generateTokenIds(prompts, samplingParams, false, Duration.ZERO, onToken);
  }

  /**
   * Same as {@link #generateTokenIds(List, SamplingParams)} with a seq-aware token callback and no
   * timeout. {@link TokenEvent#seqId()} matches the prompt index.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param onToken        per-token events ({@code null} = no stream)
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final SamplingParams samplingParams,
    final Consumer<TokenEvent> onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(prompts, samplingParams, false, Duration.ZERO, onToken);
  }

  /**
   * Token-id generate with a wall-clock limit and a seq-aware token callback.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param timeout        max wall time; {@code null} / zero / negative = none
   * @param onToken        per-token events ({@code null} = no stream); must not re-enter generate
   * @return one output per prompt
   * @since 1.1.0
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final SamplingParams samplingParams,
    final Duration timeout,
    final Consumer<TokenEvent> onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(prompts, samplingParams, false, timeout, onToken);
  }

  /**
   * Completes pre-tokenized prompts with per-prompt sampling (no progress, timeout, or stream).
   * List sizes must match.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams one params object per prompt; sizes must match
   * @return one output per prompt
   * @throws IllegalArgumentException if list sizes differ
   * @since 1.1.0
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final List<SamplingParams> samplingParams
  ) {
    return this.generateTokenIds(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * Full token-id generate: enqueue → drive ticks until idle → decode completions.
   *
   * @param prompts        one {@link List} of token ids per sequence (model vocabulary); non-{@code null}
   * @param samplingParams shared sampling; non-{@code null}
   * @param useTqdm        progress when listener is not silent
   * @param timeout        wall-clock limit; {@code null}/zero/negative = none
   * @param onToken        optional streaming callback; must not re-enter generate
   * @return one output per prompt
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws GenerationTimeoutException   if {@code timeout} elapses
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   * @apiNote Same exclusivity and {@code onToken} rules as {@link #generate(List, SamplingParams, boolean, Duration, IntConsumer)}.
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final SamplingParams samplingParams,
    final boolean useTqdm,
    final Duration timeout,
    final IntConsumer onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(
      prompts, samplingParams, useTqdm, timeout, LLM.adaptTokenCallback(onToken));
  }

  /**
   * Token-id generate with per-prompt sampling parameters.
   *
   * @param prompts        one token-id list per sequence; non-{@code null}
   * @param samplingParams one params object per prompt; sizes must match; non-{@code null}
   * @param useTqdm        progress when listener is not silent
   * @param timeout        wall-clock limit; {@code null}/zero/negative = none
   * @param onToken        optional streaming callback; must not re-enter generate
   * @return one output per prompt
   * @throws IllegalArgumentException     if list sizes differ
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws GenerationTimeoutException   if {@code timeout} elapses
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   */
  public List<GenerationOutput> generateTokenIds(
    final List<List<Integer>> prompts,
    final List<SamplingParams> samplingParams,
    final boolean useTqdm,
    final Duration timeout,
    final IntConsumer onToken
  ) {
    requireNonNull(prompts, "prompts");
    requireNonNull(samplingParams, "samplingParams");
    return this.generateUntyped(
      prompts, samplingParams, useTqdm, timeout, LLM.adaptTokenCallback(onToken));
  }

  private List<GenerationOutput> generateUntyped(
    final List<?> prompts,
    final Object samplingParams,
    final boolean useTqdm,
    final Duration timeout,
    final Consumer<TokenEvent> onToken
  ) {
    // Business: turn prompts into decoded completions under cancel / timeout / optional streaming
    this.assertNotClosed();
    this.generateLock.lock();
    try {
      this.assertNotClosed();
      // Reset cancel flag for this exclusive generate session
      this.beginGeneration();

      // Normalize sampling (shared or per-prompt) and enqueue every prompt as a Sequence
      List<SamplingParams> params = this.resolveSamplingParams(prompts, samplingParams);
      this.enqueueAllPromptsUnlocked(prompts, params);

      // Wall-clock budget and UI progress knobs for the drive loop
      long startedAtNanos = System.nanoTime();
      long deadlineNanos = this.resolveDeadlineNanos(timeout, startedAtNanos);
      boolean showProgress = this.shouldShowProgress(useTqdm);
      Map<Integer, FinishedOutput> outputsBySeqId = new HashMap<>();

      try {
        // Core loop: step until the scheduler has no waiting/running work
        this.driveUntilSchedulerIdle(
          timeout,
          deadlineNanos,
          onToken,
          showProgress,
          prompts.size(),
          startedAtNanos,
          outputsBySeqId);
      } finally {
        // On cancel/timeout/error: abort leftover sequences; always clear cancel flag
        this.finishGeneration();
      }

      // End the progress line (if any) and decode completion token ids → text
      this.finishProgressLine(showProgress);
      long elapsedNanos = Math.max(0L, System.nanoTime() - startedAtNanos);
      return this.decodeCompletedOutputs(outputsBySeqId, elapsedNanos);
    } finally {
      this.generateLock.unlock();
    }
  }

  private void beginGeneration() {
    // Internal: this generate owns the cancel flag until finishGeneration
    this.cancelRequested.set(false);
  }

  private void finishGeneration() {
    // Internal: incomplete work after abort must free KV pages via scheduler.clear()
    if (!this.scheduler.isFinished()) {
      this.scheduler.clear();
    }
    this.cancelRequested.set(false);
  }

  private void enqueueAllPromptsUnlocked(final List<?> prompts, final List<SamplingParams> params) {
    // Internal: one Sequence per prompt, paired with its SamplingParams (caller holds generateLock)
    IntStream.range(0, prompts.size())
      .forEach(i -> this.enqueuePromptUnlocked(prompts.get(i), params.get(i)));
  }

  private void enqueuePromptUnlocked(final Object prompt, final SamplingParams samplingParams) {
    // Internal: accept either raw text (tokenize) or pre-tokenized ids
    if (prompt instanceof String text) {
      this.enqueueTokenIdsUnlocked(this.tokenizer.encode(text), samplingParams);
      return;
    }
    if (prompt instanceof List<?> ids) {
      this.enqueueTokenIdsUnlocked(this.toTokenIdList(ids), samplingParams);
      return;
    }
    throw new IllegalArgumentException("prompt must be String or List<Integer>");
  }

  private void enqueueTokenIdsUnlocked(
    final List<Integer> tokenIds,
    final SamplingParams samplingParams
  ) {
    this.scheduler.add(new Sequence(
      tokenIds,
      this.fitSamplingToContext(tokenIds.size(), samplingParams),
      this.config.kvcacheBlockSize()));
  }

  private SamplingParams fitSamplingToContext(final int promptLen, final SamplingParams params) {
    int maxLen = this.config.maxModelLen();
    if (promptLen >= maxLen) {
      throw new IllegalArgumentException(
        "prompt length %d exceeds maxModelLen %d (no room for generation)"
          .formatted(promptLen, maxLen));
    }
    int room = maxLen - promptLen;
    return params.maxTokens() <= room ? params : params.withMaxTokens(room);
  }

  private List<Integer> toTokenIdList(final List<?> ids) {
    // Internal: untyped List<?> from the public generate API → List<Integer>
    return ids.stream()
      .map(id -> ((Number) id).intValue())
      .toList();
  }

  private void driveUntilSchedulerIdle(
    final Duration timeout,
    final long deadlineNanos,
    final Consumer<TokenEvent> onToken,
    final boolean showProgress,
    final int totalPrompts,
    final long startedAtNanos,
    final Map<Integer, FinishedOutput> outputsBySeqId
  ) {
    // Business loop: keep producing tokens until every sequence is finished or aborted
    int completed = 0;
    while (!this.scheduler.isFinished()) {
      // Guard: cancel flag or wall-clock deadline → throw (finally clears the scheduler)
      this.requireGenerationAllowed(timeout, deadlineNanos);

      // One engine tick (schedule → forward+sample → postprocess); lock already held
      StepResult step = this.stepUnlocked();

      // Optional streaming: notify caller for each newly appended token id
      this.dispatchTokenEvents(step, onToken);

      // Collect finished sequences; update progress counter when a prompt completes
      completed = this.recordFinishedOutputs(
        step, outputsBySeqId, completed, showProgress, totalPrompts, startedAtNanos);
    }
  }

  private void requireGenerationAllowed(final Duration timeout, final long deadlineNanos) {
    // Internal: both guards must pass before the next tick
    this.requireNotCancelled();
    this.requireWithinDeadline(timeout, deadlineNanos);
  }

  private void requireNotCancelled() {
    // Internal: cancel() from another thread sets this flag
    if (this.cancelRequested.get()) {
      throw new GenerationCancelledException();
    }
  }

  private void requireWithinDeadline(final Duration timeout, final long deadlineNanos) {
    // Internal: Long.MAX_VALUE means “no timeout”
    if (System.nanoTime() > deadlineNanos) {
      throw new GenerationTimeoutException(timeout);
    }
  }

  private void dispatchTokenEvents(final StepResult step, final Consumer<TokenEvent> onToken) {
    // Internal: skip when the caller did not request streaming
    if (onToken == null) {
      return;
    }
    step.tokenEvents().forEach(onToken);
  }

  private int recordFinishedOutputs(
    final StepResult step,
    final Map<Integer, FinishedOutput> outputsBySeqId,
    final int completed,
    final boolean showProgress,
    final int totalPrompts,
    final long startedAtNanos
  ) {
    // Internal: stash finished sequences by seqId; progress counts finished prompts
    int nextCompleted = completed;
    for (FinishedOutput finished : step.outputs()) {
      outputsBySeqId.put(finished.seqId(), finished);
      nextCompleted++;
      this.reportProgressIfNeeded(showProgress, nextCompleted, totalPrompts, startedAtNanos);
    }
    return nextCompleted;
  }

  private void reportProgressIfNeeded(
    final boolean showProgress,
    final int completed,
    final int totalPrompts,
    final long startedAtNanos
  ) {
    // Internal: CLI-style progress only when requested and listener is not silent
    if (!showProgress) {
      return;
    }
    double elapsedSeconds = (System.nanoTime() - startedAtNanos) / 1e9;
    LlmListeners.progressf(this.listener, this, "\rGenerating %d/%d (%.1fs)", completed,
      totalPrompts, elapsedSeconds);
  }

  private void finishProgressLine(final boolean showProgress) {
    // Internal: move past the \r progress line before returning text
    if (showProgress) {
      LlmListeners.progressf(this.listener, this, "%n");
    }
  }

  private long resolveDeadlineNanos(final Duration timeout, final long startedAtNanos) {
    // Internal: null / zero / negative Duration → unbounded generate
    if (timeout == null || timeout.isZero() || timeout.isNegative()) {
      return Long.MAX_VALUE;
    }
    return startedAtNanos + timeout.toNanos();
  }

  private boolean shouldShowProgress(final boolean useTqdm) {
    // Internal: never spam progress into silent (library-default) I/O
    return useTqdm && !LlmListeners.isSilent(this.listener);
  }

  private List<GenerationOutput> decodeCompletedOutputs(
    final Map<Integer, FinishedOutput> outputsBySeqId,
    final long elapsedNanos
  ) {
    // Internal: stable seqId order → GenerationOutput(text, tokenIds, stats) for each prompt
    return outputsBySeqId.keySet().stream()
      .sorted()
      .map(seqId -> {
        FinishedOutput finished = outputsBySeqId.get(seqId);
        List<Integer> tokenIds = finished.tokenIds();
        GenerationStats stats = new GenerationStats(
          finished.promptTokenCount(), tokenIds.size(), elapsedNanos);
        return new GenerationOutput(
          this.tokenizer.decode(tokenIds, true), tokenIds, stats);
      })
      .toList();
  }

  private List<SamplingParams> resolveSamplingParams(final List<?> prompts,
                                                     final Object samplingParams) {
    // Internal: one shared SamplingParams, or one per prompt (sizes must match)
    if (samplingParams instanceof SamplingParams shared) {
      return java.util.Collections.nCopies(prompts.size(), shared);
    }
    if (samplingParams instanceof List<?> list) {
      if (list.size() != prompts.size()) {
        throw new IllegalArgumentException(
          "samplingParams list size %d must match prompts size %d"
            .formatted(list.size(), prompts.size()));
      }
      return list.stream()
        .map(SamplingParams.class::cast)
        .toList();
    }
    throw new IllegalArgumentException(
      "samplingParams must be SamplingParams or List<SamplingParams>");
  }

  /**
   * Engine sampling policy from {@link Builder#sampling(SamplingParams)}, or
   * {@link SamplingDefaults#neutral()} when unset. Chat / complete / RAG helpers use this unless
   * the call overrides max tokens or passes its own {@link SamplingParams}.
   *
   * @return a new immutable {@link SamplingParams}; never {@code null}
   * @throws IllegalStateException if this engine is closed
   */
  public SamplingParams defaultSampling() {
    this.assertNotClosed();
    return this.defaultSampling;
  }

  /**
   * {@link #defaultSampling()} with a custom max new-token budget; temperature, top-k, and top-p
   * stay unchanged. Chat / complete / RAG max-token shortcuts use this.
   *
   * @param maxTokens maximum newly generated tokens per sequence; must be {@code >= 1}
   * @return a new immutable {@link SamplingParams}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @throws IllegalStateException    if this engine is closed
   */
  public SamplingParams defaultSampling(final int maxTokens) {
    this.assertNotClosed();
    return this.defaultSampling.withMaxTokens(maxTokens);
  }

  /**
   * Opens a multi-turn {@link ChatSession} using {@link #defaultSampling()} and this instance's
   * {@link #systemPrompt()}. The session applies the tokenizer chat template, keeps history, and
   * parses thinking/answer. Not thread-safe; sends use this engine exclusively. Prefer this over
   * raw {@link #generate} for dialog.
   *
   * @return a new session; not thread-safe; uses this {@code LLM} exclusively while sending
   * @throws IllegalStateException if this engine is closed
   */
  public ChatSession chat() {
    this.assertNotClosed();
    return new ChatSession(this);
  }

  /**
   * Opens a multi-turn session with explicit sampling for each {@link ChatSession#send}. Other
   * session behavior matches {@link #chat()}.
   *
   * @param samplingParams sampling for each {@link ChatSession#send}; non-{@code null}
   * @return a new session bound to this engine
   * @throws NullPointerException  if {@code samplingParams} is {@code null}
   * @throws IllegalStateException if this engine is closed
   */
  public ChatSession chat(final SamplingParams samplingParams) {
    this.assertNotClosed();
    return new ChatSession(this, samplingParams);
  }

  /**
   * Opens a multi-turn session with engine sampling limited to {@code maxTokens}
   * ({@link #defaultSampling(int)}). Temperature / top-k / top-p stay at {@link #defaultSampling()}.
   *
   * @param maxTokens max new tokens per turn; must be {@code >= 1}
   * @return a new session
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @throws IllegalStateException    if this engine is closed
   */
  public ChatSession chat(final int maxTokens) {
    this.assertNotClosed();
    return ChatSession.open(this, maxTokens);
  }

  /**
   * Opens a retrieval-augmented session over {@code index} (typically a {@link PreparedRag} from
   * {@link RagFactory}). Each user turn retrieves passages, then chat-generates. Chunk size is
   * chosen at index load via {@link com.igormaznitsa.nanollvm.rag.RagLoadOptions}, not here.
   * Grounded turns may clamp sampling temperature; thinking is off by default on the session.
   *
   * @param index corpus index; non-{@code null}; may be shared across sessions/engines
   * @return a new {@link RagSession} using {@link #defaultSampling()}
   * @throws NullPointerException  if {@code index} is {@code null}
   * @throws IllegalStateException if this engine is closed
   * @apiNote Session turns call {@link #generate}; exclusive on this {@code LLM} while active.
   * @see com.igormaznitsa.nanollvm.rag.RagLoadOptions#withMaxChunkChars(int)
   */
  public RagSession rag(final RagIndex index) {
    this.assertNotClosed();
    return RagSession.open(this, index);
  }

  /**
   * Opens a RAG session like {@link #rag(RagIndex)} with model-aware sampling limited to
   * {@code maxTokens} (other knobs from {@link #defaultSampling()}).
   *
   * @param index     corpus index; non-{@code null}
   * @param maxTokens max new tokens per answer turn; must be {@code >= 1}
   * @return a new session
   * @throws NullPointerException     if {@code index} is {@code null}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @throws IllegalStateException    if this engine is closed
   */
  public RagSession rag(final RagIndex index, final int maxTokens) {
    this.assertNotClosed();
    return RagSession.open(this, index, maxTokens);
  }

  /**
   * Raw text continuation with {@link #defaultSampling()}. No chat template and no history —
   * the model sees {@code prompt} as a prefix to complete. For instruct/chat models prefer
   * {@link #chatOnce(String)}.
   *
   * @param prompt continuation seed as plain text; non-{@code null} (may be empty)
   * @return decoded completion text for the single prompt (not including the prompt itself)
   * @throws NullPointerException         if {@code prompt} is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws IllegalStateException        if this engine is closed
   */
  public String complete(final String prompt) {
    return this.complete(prompt, this.defaultSampling());
  }

  /**
   * Raw text continuation like {@link #complete(String)} with engine sampling limited to
   * {@code maxTokens}. Other knobs stay at {@link #defaultSampling()}.
   *
   * @param prompt    continuation seed; non-{@code null}
   * @param maxTokens new-token cap; must be {@code >= 1}
   * @return decoded completion (prompt not echoed)
   * @since 1.1.0
   */
  public String complete(final String prompt, final int maxTokens) {
    return this.complete(prompt, this.defaultSampling(maxTokens));
  }

  /**
   * Raw text completion (no chat template).
   *
   * @param prompt         continuation seed as plain text; non-{@code null}
   * @param samplingParams sampling controls; non-{@code null}
   * @return decoded completion text (prompt not echoed)
   * @throws NullPointerException         if either argument is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   */
  public String complete(final String prompt, final SamplingParams samplingParams) {
    // Business: raw continuation (no chat template) → single decoded string
    requireNonNull(prompt, "prompt");
    List<GenerationOutput> outputs = this.generate(List.of(prompt), samplingParams);
    return outputs.getFirst().text();
  }

  /**
   * Single-turn chat: system prompt (if any) + one user message, then the assistant answer text.
   * Uses {@link #defaultSampling()}. History is not retained after the call.
   *
   * @param userMessage user turn text; non-{@code null}
   * @return visible answer only (thinking / tags stripped by the session parser)
   * @throws NullPointerException         if {@code userMessage} is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   */
  public String chatOnce(final String userMessage) {
    return this.chatOnce(userMessage, this.defaultSampling());
  }

  /**
   * Single-turn chat with engine sampling limited to {@code maxTokens}. History is not retained;
   * thinking/answer parsing matches {@link #chatOnce(String)}.
   *
   * @param userMessage user turn text; non-{@code null}
   * @param maxTokens   new-token cap; must be {@code >= 1}
   * @return visible answer only
   * @since 1.1.0
   */
  public String chatOnce(final String userMessage, final int maxTokens) {
    return this.chatOnce(userMessage, this.defaultSampling(maxTokens));
  }

  /**
   * Single-turn chat with explicit sampling. History is not retained after the call.
   *
   * @param userMessage    user turn text; non-{@code null}
   * @param samplingParams sampling for this one turn; non-{@code null}
   * @return visible answer only
   * @throws NullPointerException         if either argument is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   */
  public String chatOnce(final String userMessage, final SamplingParams samplingParams) {
    // Business: one ChatSession turn; return visible answer only (thinking stripped by session)
    ChatReply reply = this.chat(samplingParams).send(userMessage);
    return reply.answer();
  }

  /**
   * Tokenizer bound to {@link #model()}. Encode/decode and chat templates go through this instance.
   * Treat it as immutable from the application; do not share mutations across engines.
   *
   * @return shared tokenizer instance; never {@code null}
   */
  public Tokenizer tokenizer() {
    return this.tokenizer;
  }

  /**
   * Scratchpad markers from {@link LlmModel#thinkTags()} — the pair belongs to the checkpoint,
   * not this engine. Override per conversation with {@link ChatSession#thinkTags(ThinkTags)}.
   * ChatML skip-seed only applies when both markers are whole vocab tokens.
   *
   * @return model think-tag pair; never {@code null}
   * @since 1.1.0
   */
  public ThinkTags thinkTags() {
    return this.model.thinkTags();
  }

  /**
   * Chat-markup search strings from {@link LlmModel#chatSpecials()} — the list belongs to the
   * checkpoint, not this engine. Used when stripping specials from visible answers.
   *
   * @return model chat specials; never {@code null}
   * @since 1.1.0
   */
  public ChatSpecials chatSpecials() {
    return this.model.chatSpecials();
  }

  /**
   * Live engine configuration owned by this {@code LLM}.
   *
   * @return sealed config from construction; never {@code null}
   * @apiNote Treat as read-only. Do not mutate (stop tokens / KV sizing are fixed at build).
   */
  public Config config() {
    return this.config;
  }

  /**
   * Status / progress / text event sink used by load messages and optional generate progress.
   * Construction defaults to {@link LlmListeners#silent()}; CLI tools typically pass
   * {@link LlmListeners#toSystem()} via {@link Builder#listen(LlmListener)}.
   *
   * @return listener from the builder; never {@code null}
   */
  public LlmListener listener() {
    return this.listener;
  }

  /**
   * System text used by {@link #newConversation()} and {@link #chat()}.
   * Builder override wins; otherwise the library default is empty (no system turn).
   * Set {@link Builder#systemPrompt(String)} for application policy.
   *
   * @return system prompt string; never {@code null} (may be blank = no system turn)
   */
  public String systemPrompt() {
    if (this.systemPromptOverride != null) {
      return this.systemPromptOverride;
    }
    return "";
  }

  /**
   * Advisors configured at build time. Empty when {@link Builder#noAdvisors()} or none were set.
   * Chat / RAG run this list as one batched generate before the main reply.
   *
   * @return immutable list; empty when advisors are disabled; never {@code null}
   */
  public List<LlmAdvisor> advisors() {
    return this.advisors;
  }

  /**
   * Mixer used by {@link #runAdvisors} after advisor generates. Defaults to
   * {@link LlmAdvisorMixer#defaults()} (insert notes into the facts block).
   *
   * @return configured mixer; never {@code null}
   */
  public LlmAdvisorMixer advisorMixer() {
    return this.advisorMixer;
  }

  /**
   * Keeps advisor notes that pass this predicate after decode (default: non-blank). Demo policies
   * may reject short setup acknowledgments before mix / salvage. Set via
   * {@link Builder#advisorNoteFilter(Predicate)}.
   *
   * @return filter used on advisor answer text; never {@code null}
   * @since 1.1.0
   */
  public Predicate<String> advisorNoteFilter() {
    return this.advisorNoteFilter;
  }

  /**
   * Runs configured advisors for one chat/RAG user turn (no-op enrichment when none configured).
   *
   * @param modelUserText  user text advisors see as the latest turn (often RAG-augmented);
   *                       non-{@code null}
   * @param priorDialog    earlier user/assistant turns for anaphora (system turns ignored);
   *                       non-{@code null}, may be empty
   * @param samplingParams sampling for advisor generates; non-{@code null}
   * @return enrichment notes for mix / thinking; never {@code null}
   * @throws NullPointerException         if any argument is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires during advisor generates
   * @apiNote Calls {@link #generate} and therefore must not be invoked from an {@code onToken}
   * callback or while already holding the generate lock.
   */
  public AdvisorEnrichment runAdvisors(
    final String modelUserText,
    final List<ChatMessage> priorDialog,
    final SamplingParams samplingParams
  ) {
    this.assertNotClosed();
    return AdvisorRunner.enrich(this, modelUserText, priorDialog, samplingParams);
  }

  /**
   * Runs advisors with no prior dialog (first turn / callers without history). Same as
   * {@link #runAdvisors(String, List, SamplingParams)} with an empty history list.
   *
   * @param modelUserText  user text advisors see as the latest turn; non-{@code null}
   * @param samplingParams sampling for advisor generates; non-{@code null}
   * @return enrichment notes for mix / thinking; never {@code null}
   * @see #runAdvisors(String, List, SamplingParams)
   */
  public AdvisorEnrichment runAdvisors(final String modelUserText,
                                       final SamplingParams samplingParams) {
    return this.runAdvisors(modelUserText, List.of(), samplingParams);
  }

  /**
   * Fresh dialog history: optional system turn from {@link #systemPrompt()}, nothing else.
   * Blank system prompt means no system message. Suitable as the seed list for a new
   * {@link ChatSession}; the session will append user/assistant turns itself.
   *
   * @return a new mutable list; never {@code null}
   * @throws IllegalStateException if this engine is closed
   */
  public List<ChatMessage> newConversation() {
    this.assertNotClosed();
    return ChatMessages.newConversation(this.systemPrompt());
  }

  /**
   * {@code true} after {@link #close()} has released per-engine resources. Further generate / chat
   * / RAG calls throw {@link IllegalStateException}. The bound {@link #model()} may still be open.
   *
   * @return whether this engine was closed
   */
  public boolean isClosed() {
    return this.closed.get();
  }

  /**
   * Cancels any in-flight generate and releases per-engine resources (scheduler, KV/conv arenas,
   * matmul runtime, engine lease on the shared model). Idempotent; does not
   * {@link LlmModel#close() close} the shared model.
   *
   * @apiNote Blocks until any in-flight {@link #generate} can be interrupted and the transformer
   * / matmul runtime are closed under the generate lock.
   */
  @Override
  public void close() {
    this.cancel();
    this.generateLock.lock();
    try {
      if (this.closed.compareAndSet(false, true)) {
        this.scheduler.clear();
        try {
          this.transformer.close();
        } finally {
          this.matmul.close();
          LlmModelImpl.peer(this.model).releaseEngine();
        }
      }
    } finally {
      this.generateLock.unlock();
    }
  }

  private void assertNotClosed() {
    if (this.closed.get()) {
      throw new IllegalStateException("LLM is closed");
    }
  }

  /**
   * Fluent configurator for {@link LLM}.
   *
   * <p>Defaults: {@link LlmListeners#silent()}, warmup off, GGUF weights packed (size-first;
   * {@link #allowUnpackParameters()} for float32 speed), empty system prompt, no advisors,
   * {@link SamplingDefaults#neutral()} until {@link #sampling(SamplingParams)}.
   *
   * <p>Provide a shared {@link LlmModel} via {@link LLM#builder(LlmModel)}. Call {@link #build()} last.
   *
   * <h2>Which setter?</h2>
   * <ul>
   *   <li><b>Reply style</b> — {@link #sampling}, {@link #systemPrompt} / {@link #noSystemPrompt},
   *       {@link #advisors}</li>
   *   <li><b>How much text one generate may hold</b> — {@link #maxModelLen}
   *       (prompt + new tokens; not the same as {@code maxTokens} on sampling)</li>
   *   <li><b>CPU / speed</b> — {@link #cpuThreads}, {@link #allCpuThreads}, {@link #disableMultiCpu},
   *       {@link #matmulExecutor}, {@link #dedicatedMatmulPool}, {@link #allowUnpackParameters},
   *       {@link #warmup}</li>
   *   <li><b>RAM for conversation memory</b> — {@link #kvHeapFraction}, {@link #numKvcacheBlocks},
   *       {@link #maxModelLen}, {@link #maxNumSeqs}</li>
   *   <li><b>When a reply must stop</b> — {@link #stopTokenIds} (plus sampling {@code maxTokens})</li>
   *   <li><b>Load / progress logs</b> — {@link #listen} ({@link LlmListeners#toSystem()} for CLI)</li>
   * </ul>
   * Leave {@link #kvcacheBlockSize} and {@link #maxNumBatchedTokens} at defaults unless you are
   * tuning memory under a large batch.
   *
   * <p><strong>Validation (enforced at {@link #build()} / {@link Config} construction):</strong>
   * {@code kvcacheBlockSize} multiple of 256; {@code cpuThreads >= 1}; {@code numKvcacheBlocks}
   * resolved to {@code > 0}. Advisor lists require a mixer plus non-blank
   * unique names ({@link #advisors(LlmAdvisorMixer, LlmAdvisor...)}).
   */
  public static final class Builder {

    private final LlmModel sharedModel;
    private LlmListener listener = LlmListeners.silent();
    private int maxNumBatchedTokens = 16384;
    private int maxNumSeqs = 512;
    private int maxModelLen = 4096;
    private float kvHeapFraction = 0.25f;
    private Integer cpuThreads;
    private ExecutorService matmulExecutor;
    private boolean dedicatedMatmulPool;
    private int kvcacheBlockSize = 256;
    private int numKvcacheBlocks = -1;
    private boolean warmup = false;
    private boolean allowUnpackParameters = false;
    /**
     * {@code null} = empty library default; blank = no system turn; non-blank = caller policy.
     */
    private String systemPromptOverride = null;
    private List<LlmAdvisor> advisors = List.of();
    private LlmAdvisorMixer advisorMixer = LlmAdvisorMixer.defaults();
    private Predicate<String> advisorNoteFilter = note -> note != null && !note.isBlank();
    private SamplingParams samplingParams;
    private List<Integer> stopTokenIds;

    private Builder(final LlmModel model) {
      this.sharedModel = requireNonNull(model, "model");
    }

    private static List<LlmAdvisor> requireUniqueAdvisors(
      final Collection<? extends LlmAdvisor> advisors
    ) {
      Set<String> seen = new HashSet<>();
      List<LlmAdvisor> copy = new ArrayList<>(advisors.size());
      for (LlmAdvisor advisor : advisors) {
        requireNonNull(advisor, "advisor");
        String name = advisor.name().strip();
        if (name.isEmpty()) {
          throw new IllegalArgumentException("advisor name must not be blank");
        }
        if (!seen.add(name.toLowerCase(Locale.ROOT))) {
          throw new IllegalArgumentException("duplicate advisor name: " + name);
        }
        copy.add(advisor);
      }
      return List.copyOf(copy);
    }

    /**
     * Status / chat event sink; {@code null} is treated as {@link LlmListeners#silent()}.
     * CLI tools typically pass {@link LlmListeners#toSystem()}.
     *
     * @param listener event sink, or {@code null} for silent
     * @return {@code this}
     */
    public Builder listen(final LlmListener listener) {
      this.listener = listener == null ? LlmListeners.silent() : listener;
      return this;
    }

    /**
     * Sets the chat system prompt. {@code null} becomes blank (no system turn).
     * Omit this method entirely to keep the empty library default.
     *
     * @param prompt system text, {@code null}/{@code ""} for no system turn, or non-blank text
     * @return {@code this}
     */
    public Builder systemPrompt(final String prompt) {
      this.systemPromptOverride = prompt == null ? "" : prompt;
      return this;
    }

    /**
     * Forces no system turn in {@link LLM#newConversation()} / chat helpers. Equivalent to
     * {@link #systemPrompt(String) systemPrompt("")}. Use when the checkpoint template should not
     * see a system message.
     *
     * @return {@code this}
     */
    public Builder noSystemPrompt() {
      return this.systemPrompt("");
    }

    /**
     * Clears a previous {@link #systemPrompt(String)} / {@link #noSystemPrompt()} override so the
     * empty library default applies again (no system turn unless you set one).
     *
     * @return {@code this}
     */
    public Builder defaultSystemPrompt() {
      this.systemPromptOverride = null;
      return this;
    }

    /**
     * Configures named advisors and how their replies are mixed into the main user prompt.
     * Advisors run as one batched {@link LLM#generate}. Empty advisor varargs clear advisors.
     *
     * @param mixer    mixes advisor replies into the turn prompt; non-{@code null}
     * @param advisors advisors with non-blank unique names; non-{@code null} elements
     * @return {@code this}
     * @throws NullPointerException     if {@code mixer}, {@code advisors}, or an element is {@code null}
     * @throws IllegalArgumentException if names are blank or not unique (case-insensitive)
     */
    public Builder advisors(final LlmAdvisorMixer mixer, final LlmAdvisor... advisors) {
      requireNonNull(mixer, "mixer");
      requireNonNull(advisors, "advisors");
      return this.advisors(mixer, Arrays.asList(advisors));
    }

    /**
     * Configures named advisors and how their replies are mixed into the main user prompt.
     * Advisors run as one batched {@link LLM#generate}. An empty collection clears advisors.
     *
     * @param mixer    mixes advisor replies into the turn prompt; non-{@code null}
     * @param advisors advisors with non-blank unique names; non-{@code null} elements
     * @return {@code this}
     * @throws NullPointerException     if {@code mixer}, {@code advisors}, or an element is {@code null}
     * @throws IllegalArgumentException if names are blank or not unique (case-insensitive)
     */
    public Builder advisors(
      final LlmAdvisorMixer mixer,
      final Collection<? extends LlmAdvisor> advisors
    ) {
      requireNonNull(mixer, "mixer");
      requireNonNull(advisors, "advisors");
      this.advisorMixer = mixer;
      if (advisors.isEmpty()) {
        this.advisors = List.of();
        return this;
      }
      this.advisors = Builder.requireUniqueAdvisors(advisors);
      return this;
    }

    /**
     * Clears advisor configuration so chat / RAG turns skip the advisor pass (no extra batched
     * generate). Mixer resets to {@link LlmAdvisorMixer#defaults()}.
     *
     * @return {@code this}
     */
    public Builder noAdvisors() {
      this.advisors = List.of();
      this.advisorMixer = LlmAdvisorMixer.defaults();
      return this;
    }

    /**
     * Predicate for keeping advisor note text after decode (default: non-blank). Applications may
     * reject demo setup fillers before mix / salvage. Does not change whether advisors run.
     *
     * @param filter keep-test; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException if {@code filter} is {@code null}
     * @since 1.1.0
     */
    public Builder advisorNoteFilter(final Predicate<String> filter) {
      this.advisorNoteFilter = requireNonNull(filter, "filter");
      return this;
    }

    /**
     * Seals default sampling for {@link LLM#chat()}, {@link LLM#chatOnce(String)},
     * {@link LLM#complete(String)}, {@link LLM#rag(RagIndex)}, and {@link LLM#defaultSampling()}.
     * Unset → {@link SamplingDefaults#neutral()}. {@link LLM#chat(int)} / max-token shortcuts
     * override only {@code maxTokens}. See {@link SamplingParams} for temperature, top-k, and
     * top-p (lower temperature → less random; greedy {@code 0} is rejected).
     *
     * @param samplingParams default knobs; must not be {@code null}
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder sampling(final SamplingParams samplingParams) {
      this.samplingParams = requireNonNull(samplingParams, "samplingParams");
      return this;
    }

    /**
     * Extra tokens that should end a reply (besides the checkpoint's own end marker).
     * Replaces the tokenizer stop list after it is applied. First id is also the engine EOS.
     * Use when leftover chat markers appear in answers. Must be non-empty.
     *
     * @param ids non-empty stop-token ids; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException     if {@code ids} is {@code null}
     * @throws IllegalArgumentException if {@code ids} is empty
     * @since 1.1.0
     */
    public Builder stopTokenIds(final List<Integer> ids) {
      requireNonNull(ids, "stopTokenIds");
      if (ids.isEmpty()) {
        throw new IllegalArgumentException("stopTokenIds must not be empty");
      }
      this.stopTokenIds = List.copyOf(ids);
      return this;
    }

    /**
     * Max tokens packed into one prompt-reading step across all prompts in a batch
     * (memory / throughput bound). Default {@code 16384}. Rarely needs changing for single-user chat.
     *
     * @param value positive batch token budget
     * @return {@code this}
     */
    public Builder maxNumBatchedTokens(final int value) {
      this.maxNumBatchedTokens = value;
      return this;
    }

    /**
     * How many prompts may run together in one {@link LLM#generate} batch. Default {@code 512}.
     * Single-user chat can leave this alone; lower it if a batch of long prompts runs out of RAM.
     *
     * @param value maximum concurrent prompts
     * @return {@code this}
     */
    public Builder maxNumSeqs(final int value) {
      this.maxNumSeqs = value;
      return this;
    }

    /**
     * How much conversation this engine may hold in one go: prompt tokens plus new tokens.
     * Capped by the checkpoint's own limit. Default {@code 4096}. This is <em>not</em> reply length
     * — that is {@link SamplingParams#maxTokens()}. Raise for long documents in the prompt; lower
     * to save RAM.
     *
     * @param value context length in tokens
     * @return {@code this}
     */
    public Builder maxModelLen(final int value) {
      this.maxModelLen = value;
      return this;
    }

    /**
     * Share of JVM heap reserved for the engine's working memory of recent tokens (KV cache)
     * when {@link #numKvcacheBlocks(int)} is left on auto ({@code -1}). Default {@code 0.25}.
     * Lower if weights already fill the heap; raise only when you still have free RAM and hit
     * cache-capacity errors on long chats.
     *
     * @param value fraction in {@code (0, 1]}
     * @return {@code this}
     */
    public Builder kvHeapFraction(final float value) {
      this.kvHeapFraction = value;
      return this;
    }

    /**
     * CPU workers for dense kernels (matmul, attention, RoPE, embed gather). {@code 1} is
     * sequential (calling thread only; no executor created).
     *
     * <p>When this setter (or {@link #disableMultiCpu()} / {@link #allCpuThreads()}) is used, the
     * value wins. Otherwise {@code -Dnanollvm.cpu.threads=N} applies, else
     * {@link Runtime#availableProcessors()}. Caps how many matmul / attention / RoPE /
     * embed-gather chunks this {@code LLM} submits; the underlying pool is
     * {@link #matmulExecutor}, {@link #dedicatedMatmulPool}, or the shared lazy default
     * (only when workers &gt; 1).
     *
     * @param value worker count; must be {@code >= 1}
     * @return {@code this}
     * @throws IllegalArgumentException if {@code value < 1}
     */
    public Builder cpuThreads(final int value) {
      if (value < 1) {
        throw new IllegalArgumentException("cpuThreads must be >= 1, got " + value);
      }
      this.cpuThreads = value;
      return this;
    }

    /**
     * Sets matmul workers to {@link Runtime#availableProcessors()}. This wins over
     * {@code -Dnanollvm.cpu.threads}. For a single-thread engine use {@link #disableMultiCpu()}.
     *
     * @return {@code this}
     */
    public Builder allCpuThreads() {
      return this.cpuThreads(Runtime.getRuntime().availableProcessors());
    }

    /**
     * Sequential dense kernels on the calling thread only ({@code cpuThreads(1)}).
     * No matmul {@link ExecutorService} is created or used.
     *
     * @return {@code this}
     */
    public Builder disableMultiCpu() {
      return this.cpuThreads(1);
    }

    /**
     * Executor for parallel dense kernels (matmul, attention, RoPE, embed gather). Not shut down
     * when this {@code LLM} closes — the caller owns its lifecycle. Used only when resolved
     * {@link #cpuThreads(int)} is &gt; 1; ignored for {@link #disableMultiCpu()} / sequential.
     * When omitted and workers &gt; 1, a process-wide pool is created lazily on first parallel use
     * unless {@link #dedicatedMatmulPool()} was set. Cannot be combined with
     * {@link #dedicatedMatmulPool()}.
     *
     * @param executor non-{@code null} shared or dedicated pool
     * @return {@code this}
     * @throws NullPointerException if {@code executor} is {@code null}
     */
    public Builder matmulExecutor(final ExecutorService executor) {
      this.matmulExecutor = requireNonNull(executor, "matmulExecutor");
      return this;
    }

    /**
     * Creates a bounded pool of {@link #cpuThreads(int)} daemon workers owned by this engine.
     * {@link LLM#close()} shuts it down. Does not join the process-wide shared matmul pool.
     * Use this in a server when you want parallel matmul without handing the library a process
     * executor and without sharing {@code nanollvm-matmul-*} threads with other engines.
     * No-op when {@link #disableMultiCpu()} / {@code cpuThreads(1)}. Cannot be combined with
     * {@link #matmulExecutor(ExecutorService)}.
     *
     * @return {@code this}
     * @since 1.1.1
     */
    public Builder dedicatedMatmulPool() {
      this.dedicatedMatmulPool = true;
      return this;
    }

    /**
     * Page size of that working-memory cache, in tokens; must be a multiple of 256.
     * Default {@code 256}. Leave at default unless you are tuning paging.
     *
     * @param value block size; validated at {@link #build()}
     * @return {@code this}
     */
    public Builder kvcacheBlockSize(final int value) {
      this.kvcacheBlockSize = value;
      return this;
    }

    /**
     * How many working-memory pages to allocate. {@code -1} (default) sizes from heap and
     * {@link #kvHeapFraction(float)}. Set an explicit count only when auto-size is wrong for
     * your process. Build fails if the resolved count is not {@code > 0}.
     *
     * @param value explicit block count, or {@code -1} to auto-size
     * @return {@code this}
     */
    public Builder numKvcacheBlocks(final int value) {
      this.numKvcacheBlocks = value;
      return this;
    }

    /**
     * Expands GGUF packed weights to float32 for denser/faster matmul (higher heap).
     * Prefer {@link LlmModelFactory#open(Path)} {@code .unpackParameters()} so unpack
     * happens <em>during load</em> (file bytes → float32, no packed heap copy). For an already-loaded
     * packed {@link LlmModel}, unpacks at engine build and releases packed bytes.
     * No-op for already-dense HF safetensors. Default is packed (size-first).
     *
     * @return {@code this}
     */
    public Builder allowUnpackParameters() {
      return this.allowUnpackParameters(true);
    }

    /**
     * When {@code true}, GGUF weights are expanded to float32. With a shared packed
     * {@link LlmModel}, unpack happens at engine build (releasing packed bytes). Prefer unpacking
     * at load via {@link LlmModelFactory#make(Path, LlmListener, boolean)}. Default {@code false}.
     *
     * @param value {@code true} to unpack; {@code false} to keep GGUF packed
     * @return {@code this}
     */
    public Builder allowUnpackParameters(final boolean value) {
      this.allowUnpackParameters = value;
      return this;
    }

    /**
     * Runs a short synthetic generate after {@link #build()} to warm JIT / caches. Off by default
     * so construction stays cheap; enable for long-running servers where the first real request
     * should not pay the cold penalty.
     *
     * @return {@code this}
     */
    public Builder warmup() {
      return this.warmup(true);
    }

    /**
     * Enables or disables post-load warmup. Default {@code false}. When true, {@link #build()}
     * runs a short synthetic generate before returning.
     *
     * @param value {@code true} to run warmup inside {@link #build()}
     * @return {@code this}
     */
    public Builder warmup(final boolean value) {
      this.warmup = value;
      return this;
    }

    /**
     * Disables post-load warmup (same as the builder default). Explicit when a chain previously
     * called {@link #warmup()}.
     *
     * @return {@code this}
     */
    public Builder skipWarmup() {
      return this.warmup(false);
    }

    /**
     * Returns a ready {@link LLM} bound to the resolved {@link LlmModel}.
     *
     * @return new engine instance; caller should {@link LLM#close()} when finished
     * @throws ModelLoadException       if the model directory, config, or weights are unusable
     * @throws IllegalArgumentException if builder/config constraints fail (bad KV block size, …)
     * @throws IllegalStateException    if the model is closed or is an embedding checkpoint, or if
     *                                  {@link #matmulExecutor(ExecutorService)} and
     *                                  {@link #dedicatedMatmulPool()} were both set
     */
    public LLM build() {
      if (this.sharedModel.isClosed()) {
        throw new IllegalStateException("LlmModel is closed");
      }
      if (this.sharedModel.isEmbeddingModel()) {
        throw new IllegalStateException(
          ModelSupport.chatMisuseMessage(this.sharedModel.architectureName()));
      }
      if (this.matmulExecutor != null && this.dedicatedMatmulPool) {
        throw new IllegalStateException("cannot combine matmulExecutor with dedicatedMatmulPool");
      }
      if (!this.advisors.isEmpty()) {
        this.advisors = Builder.requireUniqueAdvisors(this.advisors);
      }
      return new LLM(this);
    }

    private LlmModel resolveModel() {
      return this.sharedModel;
    }

    private Path modelPath() {
      return this.sharedModel.path();
    }

    private Config toConfig(
      final LlmModel model,
      final Tokenizer tokenizer,
      final int cpuThreads) {
      Config.Builder config = Config.builder(model)
        .maxNumBatchedTokens(this.maxNumBatchedTokens)
        .maxNumSeqs(this.maxNumSeqs)
        .maxModelLen(this.maxModelLen)
        .kvHeapFraction(this.kvHeapFraction)
        .cpuThreads(cpuThreads)
        .kvcacheBlockSize(this.kvcacheBlockSize)
        .numKvcacheBlocks(this.numKvcacheBlocks)
        .applyTokenizer(tokenizer);
      if (this.stopTokenIds != null) {
        config.stopTokenIds(this.stopTokenIds);
      }
      return config.build();
    }

    private int resolveCpuThreads() {
      if (this.cpuThreads != null) {
        return this.cpuThreads;
      }
      String prop = NanoLlvmProps.systemProperty(NanoLlvmProps.PROP_CPU_THREADS);
      if (prop != null && !prop.isBlank()) {
        int parsed = Integer.parseInt(prop.strip());
        if (parsed < 1) {
          throw new IllegalArgumentException(
            "-" + NanoLlvmProps.PROP_CPU_THREADS + " must be >= 1, got " + parsed);
        }
        return parsed;
      }
      return Runtime.getRuntime().availableProcessors();
    }
  }

  private record FinishedOutput(int seqId, List<Integer> tokenIds, int promptTokenCount) {
    private FinishedOutput {
      tokenIds = List.copyOf(requireNonNull(tokenIds, "tokenIds"));
      if (promptTokenCount < 0) {
        throw new IllegalArgumentException("promptTokenCount must be >= 0");
      }
    }
  }

  /**
   * One newly decoded token in a {@link LLM#generate} batch. {@code seqId} is the prompt index in
   * that call (0-based). {@code tokenId} is a vocabulary id — decode with {@link #tokenizer()} if
   * you need text. Callbacks must not call generate / chat / advisors on this engine.
   *
   * @param seqId   sequence index in the current generate batch
   * @param tokenId vocabulary id of the appended token
   * @since 1.1.0
   */
  public record TokenEvent(int seqId, int tokenId) {
  }

  private record StepResult(List<FinishedOutput> outputs, List<TokenEvent> tokenEvents,
                            int numTokens) {
    private StepResult {
      outputs = List.copyOf(requireNonNull(outputs, "outputs"));
      tokenEvents = List.copyOf(requireNonNull(tokenEvents, "tokenEvents"));
    }
  }

  /**
   * Decoded completion, token ids, and engine stats for one finished prompt in
   * {@link LLM#generate}.
   *
   * <p>One output per input prompt, in the same order. {@link #text()} is a raw decode of
   * {@link #tokenIds()} — it is <em>not</em> split into thinking / answer; use
   * {@link com.igormaznitsa.nanollvm.chat.ChatReply#parse} or
   * {@link com.igormaznitsa.nanollvm.chat.ChatSession#send} for that. {@link #tokenIds()} is
   * completion tokens only (not the prompt). Immutable; lists are copies.
   *
   * @param text     tokenizer decode of {@code tokenIds} ({@code null} coerced to {@code ""})
   * @param tokenIds completion token ids only; unmodifiable copy stored
   * @param stats    prompt/completion counts and generate wall time; {@link GenerationStats#NONE}
   *                 if the caller passed {@code null}
   */
  public record GenerationOutput(String text, List<Integer> tokenIds, GenerationStats stats) {
    public GenerationOutput {
      text = text == null ? "" : text;
      tokenIds = List.copyOf(requireNonNull(tokenIds, "tokenIds"));
      stats = stats == null ? GenerationStats.NONE : stats;
    }
  }
}
