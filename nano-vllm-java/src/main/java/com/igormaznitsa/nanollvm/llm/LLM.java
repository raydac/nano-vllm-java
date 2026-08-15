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
 * Loaded causal LM for offline inference and embedding in applications.
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
 * <h2>Layers (which API?)</h2>
 * <ul>
 *   <li>{@link #chat()} / {@link #chatOnce(String)} — chat template, history, reply parsing</li>
 *   <li>{@link #complete(String)} — raw continuation of a prompt string (no chat template)</li>
 *   <li>{@link #generate(List, SamplingParams)} — simplest batch text generate</li>
 *   <li>{@link #generate(List, SamplingParams, Duration)} / {@link #generate(List, SamplingParams, Consumer)} —
 *       timeout or seq-aware token stream without dummy flags</li>
 *   <li>{@link #generate(List, SamplingParams, boolean, Duration, IntConsumer)} —
 *       full text generate (progress, timeout, token-id stream)</li>
 *   <li>{@link #generateTokenIds(List, SamplingParams)} — pre-tokenized batch</li>
 *   <li>{@link #rag(RagIndex)} — retrieval over a shared {@link PreparedRag} (or any index)</li>
 * </ul>
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
    try {
      // Business load path: resolve LlmModel → matmul runtime → engine config → KV/scheduler
      int cpuThreads = builder.resolveCpuThreads();
      MatmulRuntime.Builder matmulBuilder = MatmulRuntime.builder().cpuThreads(cpuThreads);
      if (cpuThreads > 1 && builder.matmulExecutor != null) {
        matmulBuilder.executor(builder.matmulExecutor);
      }
      createdMatmul = matmulBuilder.build();
      this.matmul = createdMatmul;
      this.model = builder.resolveModel();
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
      this.transformer = new Transformer(
        this.model, this.config, this.matmul, this.listener, builder.allowUnpackParameters);
      this.scheduler = new Scheduler(this.config, this.transformer::clearConvState);
      LlmListeners.info(this.listener, this, "CPU matmul: " + this.matmul.backendInfo());
      LlmListeners.info(this.listener, this, this.model.hasPackedWeights()
        ? "Weights: GGUF packed (specialized LinearKernel / EmbeddingKernel per tensor)"
        : "Weights: float32 dense (decode-1 + parallel MatmulRuntime)");
    } catch (ModelLoadException e) {
      this.releaseOwnedRuntime(createdMatmul);
      throw e;
    } catch (RuntimeException e) {
      this.releaseOwnedRuntime(createdMatmul);
      throw new ModelLoadException("failed to load model from " + builder.modelPath(), e);
    }
    if (builder.warmup) {
      this.warmup();
    }
  }

  /**
   * Starts a fluent configurator for a shared immutable {@link LlmModel}.
   *
   * @param model loaded model to bind; not closed by this {@code LLM}; must be non-{@code null}
   * @return a new builder; call {@link Builder#build()} to construct the engine
   * @throws NullPointerException if {@code model} is {@code null}
   */
  public static Builder builder(final LlmModel model) {
    return new Builder(requireNonNull(model, "model"));
  }

  private void releaseOwnedRuntime(final MatmulRuntime createdMatmul) {
    if (createdMatmul != null) {
      createdMatmul.close();
    }
  }

  /**
   * The immutable loaded model bound to this engine (safe to share with other {@code LLM}s).
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
      new SamplingParams(0.6f, WARMUP_DECODE_TOKENS, true),
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
   * Generates completions for one or more text prompts (no progress line, no timeout, no token callback).
   *
   * @param prompts        one string per sequence; non-{@code null}; elements non-{@code null};
   *                       size limited by {@link Config#maxNumSeqs()}
   * @param samplingParams shared sampling for every prompt; non-{@code null}
   * @return one {@link GenerationOutput} per prompt, in completion order by sequence id
   * @throws GenerationCancelledException if {@link #cancel()} fires during the run
   * @throws NullPointerException         if {@code prompts} or {@code samplingParams} is {@code null}
   * @throws IllegalArgumentException     if a prompt element is not a {@link String} (internal path)
   * @apiNote Exclusive with other generate/chat/RAG/complete on this instance.
   */
  public List<GenerationOutput> generate(final List<String> prompts,
                                         final SamplingParams samplingParams) {
    return this.generate(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * Text-prompt generate with a wall-clock limit and no token callback.
   *
   * @param timeout {@code null} / zero / negative = no limit
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
   * Text-prompt generate with a per-token id callback and no timeout.
   *
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
   * Text-prompt generate with a seq-aware token callback and no timeout.
   *
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
   * Text-prompt generate with a wall-clock limit and a seq-aware token callback.
   *
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
   * Generates completions for text prompts, optionally printing batch progress.
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
   * Text-prompt generate with per-prompt sampling (no progress, no timeout, no callback).
   *
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
   * Generates completions for pre-tokenized prompts (no progress, no timeout, no callback).
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
   * Token-id generate with a wall-clock limit and no token callback.
   *
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
   * Token-id generate with a per-token id callback and no timeout.
   *
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
   * Token-id generate with a seq-aware token callback and no timeout.
   *
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
   * Token-id generate with per-prompt sampling (no progress, no timeout, no callback).
   *
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
   * {@link SamplingDefaults#neutral()} when unset.
   *
   * @return a new immutable {@link SamplingParams}; never {@code null}
   */
  public SamplingParams defaultSampling() {
    this.assertNotClosed();
    return this.defaultSampling;
  }

  /**
   * Engine sampling with a custom max new-token budget (other knobs unchanged).
   *
   * @param maxTokens maximum newly generated tokens per sequence; must be {@code >= 1}
   * @return a new immutable {@link SamplingParams}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   */
  public SamplingParams defaultSampling(final int maxTokens) {
    this.assertNotClosed();
    return this.defaultSampling.withMaxTokens(maxTokens);
  }

  /**
   * Opens a multi-turn {@link ChatSession} using {@link #defaultSampling()} and this instance's
   * {@link #systemPrompt()}.
   *
   * @return a new session; not thread-safe; uses this {@code LLM} exclusively while sending
   */
  public ChatSession chat() {
    this.assertNotClosed();
    return new ChatSession(this);
  }

  /**
   * Opens a multi-turn session with explicit sampling.
   *
   * @param samplingParams sampling for each {@link ChatSession#send}; non-{@code null}
   * @return a new session bound to this engine
   * @throws NullPointerException if {@code samplingParams} is {@code null}
   */
  public ChatSession chat(final SamplingParams samplingParams) {
    this.assertNotClosed();
    return new ChatSession(this, samplingParams);
  }

  /**
   * Opens a multi-turn session with engine sampling limited to {@code maxTokens}
   * ({@link #defaultSampling(int)}).
   *
   * @param maxTokens max new tokens per turn; must be {@code >= 1}
   * @return a new session
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   */
  public ChatSession chat(final int maxTokens) {
    this.assertNotClosed();
    return ChatSession.open(this, maxTokens);
  }

  /**
   * Opens a retrieval-augmented session over {@code index}
   * (typically a {@link PreparedRag} from {@link RagFactory}).
   *
   * @param index corpus index; non-{@code null}; may be shared across sessions/engines
   * @return a new {@link RagSession} using {@link #defaultSampling()}
   * @throws NullPointerException if {@code index} is {@code null}
   * @apiNote Session turns call {@link #generate}; exclusive on this {@code LLM} while active.
   */
  public RagSession rag(final RagIndex index) {
    this.assertNotClosed();
    return RagSession.open(this, index);
  }

  /**
   * Opens a RAG session with model-aware sampling limited to {@code maxTokens}.
   *
   * @param index     corpus index; non-{@code null}
   * @param maxTokens max new tokens per answer turn; must be {@code >= 1}
   * @return a new session
   * @throws NullPointerException     if {@code index} is {@code null}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   */
  public RagSession rag(final RagIndex index, final int maxTokens) {
    this.assertNotClosed();
    return RagSession.open(this, index, maxTokens);
  }

  /**
   * Raw text completion (no chat template). Uses {@link #defaultSampling()}.
   *
   * @param prompt continuation seed as plain text; non-{@code null} (may be empty)
   * @return decoded completion text for the single prompt (not including the prompt itself)
   * @throws NullPointerException         if {@code prompt} is {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   */
  public String complete(final String prompt) {
    return this.complete(prompt, this.defaultSampling());
  }

  /**
   * Raw text completion with engine sampling limited to {@code maxTokens}.
   *
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
   * Single-turn chat with engine sampling limited to {@code maxTokens}. History is not retained.
   *
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
   * Tokenizer bound to {@link #model()}.
   *
   * @return shared tokenizer instance; never {@code null}; treat as immutable for callers
   */
  public Tokenizer tokenizer() {
    return this.tokenizer;
  }

  /**
   * Scratchpad markers from {@link LlmModel#thinkTags()} — the pair belongs to the checkpoint,
   * not this engine. Override per conversation with {@link ChatSession#thinkTags(ThinkTags)}.
   *
   * @since 1.1.0
   */
  public ThinkTags thinkTags() {
    return this.model.thinkTags();
  }

  /**
   * Chat-markup search strings from {@link LlmModel#chatSpecials()} — the list belongs to the
   * checkpoint, not this engine.
   *
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
   * Status / progress / text event sink used by load and optional generate progress.
   *
   * @return listener from the builder; never {@code null} ({@link LlmListeners#silent()} if unset)
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
   * Advisors configured at build time.
   *
   * @return immutable list; empty when advisors are disabled; never {@code null}
   */
  public List<LlmAdvisor> advisors() {
    return this.advisors;
  }

  /**
   * Mixer used by {@link #runAdvisors} after advisor generates.
   *
   * @return configured mixer; never {@code null}
   */
  public LlmAdvisorMixer advisorMixer() {
    return this.advisorMixer;
  }

  /**
   * Keeps advisor notes that pass this predicate (default: non-blank). Demo policies may reject
   * short setup acknowledgments.
   *
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
   * Runs advisors with no prior dialog (first turn / callers without history).
   *
   * @see #runAdvisors(String, List, SamplingParams)
   */
  public AdvisorEnrichment runAdvisors(final String modelUserText,
                                       final SamplingParams samplingParams) {
    return this.runAdvisors(modelUserText, List.of(), samplingParams);
  }

  /**
   * Fresh dialog history: optional system turn from {@link #systemPrompt()}, nothing else.
   *
   * @return a new mutable list suitable for a new {@link ChatSession}; never {@code null}
   */
  public List<ChatMessage> newConversation() {
    this.assertNotClosed();
    return ChatMessages.newConversation(this.systemPrompt());
  }

  /**
   * {@code true} after {@link #close()} has released per-engine resources.
   */
  public boolean isClosed() {
    return this.closed.get();
  }

  /**
   * Cancels any in-flight generate and releases per-engine resources (scheduler, KV/conv arenas,
   * matmul runtime mark). Idempotent; does not {@link LlmModel#close() close} the shared model.
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
        this.transformer.close();
        this.matmul.close();
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
   * <p>Defaults: {@link LlmListeners#silent()}, eager execution, warmup off, GGUF parameters
   * packed (use {@link #allowUnpackParameters()} for float32 speed), empty system prompt
   * (set {@link #systemPrompt(String)} for application policy), no advisors.
   *
   * <p>Provide a shared {@link LlmModel} via {@link LLM#builder(LlmModel)}. Call {@link #build()} last.
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
     * Forces no system turn in {@link LLM#newConversation()} / chat helpers.
     *
     * @return {@code this}
     */
    public Builder noSystemPrompt() {
      return this.systemPrompt("");
    }

    /**
     * Clears an override so the library empty default applies again.
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
     * Clears advisor configuration so chat / RAG turns skip the advisor pass.
     *
     * @return {@code this}
     */
    public Builder noAdvisors() {
      this.advisors = List.of();
      this.advisorMixer = LlmAdvisorMixer.defaults();
      return this;
    }

    /**
     * Predicate for keeping advisor note text after decode (default: non-blank).
     * Applications may reject demo setup fillers before mix / salvage.
     *
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
     * override only {@code maxTokens}.
     *
     * @since 1.1.0
     */
    public Builder sampling(final SamplingParams samplingParams) {
      this.samplingParams = requireNonNull(samplingParams, "samplingParams");
      return this;
    }

    /**
     * Replaces tokenizer EOS / stop ids after {@link Config.Builder#applyTokenizer}.
     * First id becomes {@link Config#eos()}.
     *
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
     * Max tokens across a prefill batch (scheduler / memory bound). Default {@code 16384}.
     *
     * @param value positive batch token budget
     * @return {@code this}
     */
    public Builder maxNumBatchedTokens(final int value) {
      this.maxNumBatchedTokens = value;
      return this;
    }

    /**
     * Max concurrent sequences in the scheduler. Default {@code 512}.
     * Caps how many prompts may be active in one {@link LLM#generate} batch.
     *
     * @param value maximum sequences
     * @return {@code this}
     */
    public Builder maxNumSeqs(final int value) {
      this.maxNumSeqs = value;
      return this;
    }

    /**
     * Max context length in tokens (capped by the model's {@code max_position_embeddings}).
     * Default {@code 4096}. Prompt + generated tokens should stay within this budget.
     *
     * @param value context length in tokens
     * @return {@code this}
     */
    public Builder maxModelLen(final int value) {
      this.maxModelLen = value;
      return this;
    }

    /**
     * Fraction of {@link Runtime#maxMemory()} used when estimating KV-cache size when
     * {@link #numKvcacheBlocks(int)} is {@code -1}. Default {@code 0.25}.
     *
     * @param value fraction in {@code (0, 1]}
     * @return {@code this}
     */
    public Builder kvHeapFraction(final float value) {
      this.kvHeapFraction = value;
      return this;
    }

    /**
     * CPU workers for dense matmul in {@link MatmulRuntime}. {@code 1} is sequential (calling
     * thread only; no executor created).
     *
     * <p>When this setter (or {@link #disableMultiCpu()} / {@link #allCpuThreads()}) is used, the
     * value wins. Otherwise {@code -Dnanollvm.cpu.threads=N} applies, else
     * {@link Runtime#availableProcessors()}. Caps how many matmul chunks this {@code LLM} submits;
     * the underlying pool is {@link #matmulExecutor} or the shared lazy default (only when
     * workers &gt; 1).
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
     * Sets matmul workers to {@link Runtime#availableProcessors()}.
     *
     * @return {@code this}
     */
    public Builder allCpuThreads() {
      return this.cpuThreads(Runtime.getRuntime().availableProcessors());
    }

    /**
     * Sequential dense matmul on the calling thread only ({@code cpuThreads(1)}).
     * No matmul {@link ExecutorService} is created or used.
     *
     * @return {@code this}
     */
    public Builder disableMultiCpu() {
      return this.cpuThreads(1);
    }

    /**
     * Executor for parallel dense matmul. Not shut down when this {@code LLM} closes —
     * the caller owns its lifecycle. Used only when resolved {@link #cpuThreads(int)} is &gt; 1;
     * ignored for {@link #disableMultiCpu()} / sequential. When omitted and workers &gt; 1, a
     * process-wide pool is created lazily on first parallel use.
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
     * KV block size in tokens; must be a multiple of 256. Default {@code 256}.
     *
     * @param value block size; validated at {@link #build()}
     * @return {@code this}
     */
    public Builder kvcacheBlockSize(final int value) {
      this.kvcacheBlockSize = value;
      return this;
    }

    /**
     * Number of KV blocks to allocate. {@code -1} (default) estimates from heap and config.
     * After estimation / explicit set, the effective count must be {@code > 0} or build fails.
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
     * happens <em>during load</em> (mmap → float32, no packed heap copy). For an already-loaded
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
     * Runs a short synthetic generate after load to warm JIT / caches (off by default).
     *
     * @return {@code this}
     */
    public Builder warmup() {
      return this.warmup(true);
    }

    /**
     * Enables or disables post-load warmup. Default {@code false}.
     *
     * @param value {@code true} to run warmup inside {@link #build()}
     * @return {@code this}
     */
    public Builder warmup(final boolean value) {
      this.warmup = value;
      return this;
    }

    /**
     * Disables post-load warmup (same as the builder default).
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
     */
    public LLM build() {
      if (this.sharedModel.isClosed()) {
        throw new IllegalStateException("LlmModel is closed");
      }
      if (this.sharedModel.isEmbeddingModel()) {
        throw new IllegalStateException(
          ModelSupport.chatMisuseMessage(this.sharedModel.architectureName()));
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
   * One newly decoded token in a {@link LLM#generate} batch ({@code seqId} matches prompt order).
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
