package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.engine.ModelRunner;
import com.igormaznitsa.nanollvm.engine.Scheduler;
import com.igormaznitsa.nanollvm.engine.Sequence;
import com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException;
import com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.Model;
import com.igormaznitsa.nanollvm.models.ModelFactory;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

/**
 * Loaded causal LM for offline inference and embedding in applications.
 *
 * <h2>Typical use</h2>
 * <pre>{@code
 * Model model = ModelFactory.make(modelDir);  // load once, share freely
 * try (LLM llm = LLM.builder(model)
 *         .maxModelLen(2048)
 *         .systemPrompt("Answer briefly.")  // optional
 *         .build()) {
 *     String reply = llm.chat(256).send("Hello").answer();
 *     String once = llm.chatOnce("What is 2+2?");
 *     String raw = llm.complete("The capital of France is");
 * }
 * }</pre>
 *
 * <p>Path convenience {@code LLM.builder(path)} still loads a private {@link Model} internally.
 *
 * <h2>Layers</h2>
 * <ul>
 *   <li>{@link #chat()} / {@link #chatOnce(String)} — chat template, history, reply parsing</li>
 *   <li>{@link #complete(String)} — raw continuation of a prompt string (no chat template)</li>
 *   <li>{@link #generate(List, SamplingParams)} — batch / streaming engine API</li>
 *   <li>{@link #rag(RagIndex)} — retrieval over a shared {@link PreparedRag} (or any index)</li>
 * </ul>
 *
 * <h2>Defaults</h2>
 * Construction is <em>library-quiet</em> ({@link EngineIo#silent()}). CLI tools should call
 * {@link Builder#withSystemIo()}. Architecture is auto-detected from {@code config.json}
 * (override with {@code -Dnanovllm.arch=qwen3|gemma3}).
 *
 * <h2>Thread safety</h2>
 * One instance must not run concurrent {@link #generate} or chat calls. Prefer one instance
 * per thread, or external locking. Share one immutable {@link Model} across many {@code LLM}s.
 * {@link #cancel()} is safe from another thread and aborts an in-flight generate with
 * {@link GenerationCancelledException}.
 *
 * @see Builder
 * @see Model
 * @see ModelFactory
 * @see ChatSession
 * @see EngineIo
 * @see SamplingParams
 * @see RagFactory
 * @see PreparedRag
 */
public final class LLM implements AutoCloseable {

  private static final int WARMUP_PREFILL_TOKENS = 64;
  private static final int WARMUP_DECODE_TOKENS = 16;

  private final Model model;
  private final Config config;
  private final EngineIo io;
  private final String systemPromptOverride;
  private final ModelRunner modelRunner;
  private final Tokenizer tokenizer;
  private final Scheduler scheduler;
  private final Object generateLock = new Object();
  private final AtomicBoolean cancelRequested = new AtomicBoolean();

  /**
   * Loads {@code modelPath} with default builder settings (quiet I/O, warmup on).
   *
   * @throws ModelLoadException if the model cannot be loaded
   */
  public LLM(String modelPath) {
    this(builder(modelPath));
  }

  /**
   * Loads {@code modelPath} with default builder settings (quiet I/O, warmup on).
   *
   * @throws ModelLoadException if the model cannot be loaded
   */
  public LLM(Path modelPath) {
    this(builder(modelPath));
  }

  /**
   * Binds a shared immutable {@link Model} with default builder settings.
   */
  public LLM(Model model) {
    this(builder(model));
  }

  private LLM(Builder builder) {
    requireNonNull(builder, "builder");
    try {
      // Business load path: resolve Model → engine config → KV/scheduler
      this.model = builder.resolveModel();
      this.config = builder.toConfig(this.model);
      this.io = builder.io;
      this.systemPromptOverride = builder.systemPromptOverride;
      Sequence.setBlockSize(this.config.kvcacheBlockSize());
      this.tokenizer = this.model.tokenizer();
      this.applyTokenizerStopTokens();
      this.modelRunner = new ModelRunner(this.model, this.config, this.io);
      this.scheduler = new Scheduler(this.config);
    } catch (ModelLoadException e) {
      throw e;
    } catch (RuntimeException e) {
      throw new ModelLoadException("failed to load model from " + builder.modelPath(), e);
    }
    if (builder.warmup) {
      this.warmup();
    }
  }

  private void applyTokenizerStopTokens() {
    // Internal: eos / stop ids come from the tokenizer when config did not pin them
    if (this.config.eos() < 0) {
      this.config.setEos(this.tokenizer.eosTokenId());
    }
    this.config.setStopTokenIds(this.tokenizer.stopTokenIds());
  }

  /**
   * Starts a fluent configurator for a shared immutable {@link Model}.
   */
  public static Builder builder(Model model) {
    return new Builder(requireNonNull(model, "model"));
  }

  /**
   * Starts a fluent configurator for the HuggingFace model directory.
   *
   * @param model directory containing {@code config.json}, tokenizer files, and weights
   */
  public static Builder builder(Path model) {
    return new Builder(model);
  }

  /**
   * Starts a fluent configurator for the model directory path string.
   *
   * @param modelPath path to the model directory
   */
  public static Builder builder(String modelPath) {
    return new Builder(Path.of(requireNonNull(modelPath, "modelPath")));
  }

  /**
   * The immutable loaded model bound to this engine (may be shared with other {@code LLM}s).
   */
  public Model model() {
    return this.model;
  }

  private void warmup() {
    // JIT / cache warm-up: one short generate so the first real request is not cold
    this.io.info("Warming up (prefill + decode)…");
    long startedAtNanos = System.nanoTime();
    this.generate(
        List.of(this.syntheticWarmupPrompt()),
        new SamplingParams(0.6f, WARMUP_DECODE_TOKENS, true),
        false);
    this.io.infof("Warmup done in %.1fs%n", (System.nanoTime() - startedAtNanos) / 1e9);
  }

  private List<Integer> syntheticWarmupPrompt() {
    // Fixed synthetic token ids — shape matters for warmup, not linguistic content
    return IntStream.range(0, WARMUP_PREFILL_TOKENS)
        .map(i -> 1 + (i % 97))
        .boxed()
        .toList();
  }

  /**
   * Enqueues a text prompt (tokenized with this model's tokenizer) for a later {@link #step()}.
   * Prefer {@link #generate} or {@link #chat()} unless driving the scheduler manually.
   */
  public void addRequest(String prompt, SamplingParams samplingParams) {
    // Business: string prompt → token ids → waiting queue
    this.addRequest(this.tokenizer.encode(prompt), samplingParams);
  }

  /**
   * Enqueues pre-tokenized ids for a later {@link #step()}.
   */
  public void addRequest(List<Integer> promptTokenIds, SamplingParams samplingParams) {
    // Business: register one generation sequence with the scheduler
    this.scheduler.add(new Sequence(promptTokenIds, samplingParams));
  }

  /**
   * One engine tick: schedule → forward+sample → postprocess.
   * Used by {@link #generate}; rarely needed directly.
   */
  public StepResult step() {
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

  private List<Integer> runForwardAndSample(Scheduler.ScheduleResult scheduled) {
    // Internal: CausalLM forward → logits → Sampler (one id per sequence in the batch)
    return this.modelRunner.run(scheduled.sequences(), scheduled.prefill());
  }

  private List<int[]> applySchedulerPostprocess(
      Scheduler.ScheduleResult scheduled,
      List<Integer> nextTokenIds
  ) {
    // Internal: write sampled ids into sequences; collect (seqId, tokenId) for streaming
    List<int[]> appendedTokens = new ArrayList<>();
    this.scheduler.postprocess(
        scheduled.sequences(), nextTokenIds, scheduled.prefill(), appendedTokens);
    return appendedTokens;
  }

  private List<FinishedOutput> collectFinishedOutputs(List<Sequence> sequences) {
    // Internal: only sequences that reached stop / maxTokens on this tick
    return sequences.stream()
        .filter(Sequence::isFinished)
        .map(seq -> new FinishedOutput(seq.seqId(), seq.completionTokenIds()))
        .toList();
  }

  private List<TokenEvent> toTokenEvents(List<int[]> appendedTokens) {
    // Internal: raw [seqId, tokenId] pairs → typed stream events
    return appendedTokens.stream()
        .map(pair -> new TokenEvent(pair[0], pair[1]))
        .toList();
  }

  private int measureStepWorkload(Scheduler.ScheduleResult scheduled) {
    // Internal: +token count for prefill; −batch size for decode (progress convention)
    return scheduled.prefill()
        ? scheduled.sequences().stream().mapToInt(Sequence::numScheduledTokens).sum()
        : -scheduled.sequences().size();
  }

  /**
   * Whether the scheduler has no waiting or running sequences.
   */
  public boolean isFinished() {
    return this.scheduler.isFinished();
  }

  /**
   * Requests cancellation of an in-flight {@link #generate}. Safe to call from other threads.
   * The generate call then throws {@link GenerationCancelledException}.
   */
  public void cancel() {
    this.cancelRequested.set(true);
  }

  /**
   * Generates completions for one or more prompts without progress output.
   *
   * @param prompts         each element is a {@link String} or {@code List<Integer>} token ids
   * @param samplingParams  shared sampling settings for all prompts
   * @throws GenerationCancelledException if {@link #cancel()} was called
   */
  public List<GenerationOutput> generate(List<?> prompts, SamplingParams samplingParams) {
    return this.generate(prompts, samplingParams, false, Duration.ZERO, null);
  }

  /**
   * @param useTqdm when {@code true} and I/O is not silent, prints batch progress to {@link EngineIo#out()}
   */
  public List<GenerationOutput> generate(List<?> prompts, Object samplingParams, boolean useTqdm) {
    return this.generate(prompts, samplingParams, useTqdm, Duration.ZERO, null);
  }

  /**
   * @param onToken invoked for each newly decoded token id (optional; may be {@code null})
   */
  public List<GenerationOutput> generate(
      List<?> prompts,
      Object samplingParams,
      boolean useTqdm,
      java.util.function.IntConsumer onToken
  ) {
    return this.generate(prompts, samplingParams, useTqdm, Duration.ZERO, onToken);
  }

  /**
   * Full generate entry: enqueue → drive ticks until idle → decode completions.
   *
   * @param prompts        each element is a {@link String} or {@code List<Integer>} token ids
   * @param samplingParams a {@link SamplingParams} or {@code List<SamplingParams>} (one per prompt)
   * @param useTqdm        progress line when I/O is not silent
   * @param timeout        max wall time; {@link Duration#ZERO} or negative means no limit
   * @param onToken        per-token callback, or {@code null}
   * @throws GenerationCancelledException if {@link #cancel()} fires
   * @throws GenerationTimeoutException   if {@code timeout} elapses
   * @throws IllegalArgumentException     if prompt or samplingParams types are invalid
   */
  public List<GenerationOutput> generate(
      List<?> prompts,
      Object samplingParams,
      boolean useTqdm,
      Duration timeout,
      java.util.function.IntConsumer onToken
  ) {
    // Business: turn prompts into decoded completions under cancel / timeout / optional streaming
    synchronized (this.generateLock) {
      // Reset cancel flag for this exclusive generate session
      this.beginGeneration();

      // Normalize sampling (shared or per-prompt) and enqueue every prompt as a Sequence
      List<SamplingParams> params = this.resolveSamplingParams(prompts, samplingParams);
      this.enqueueAllPrompts(prompts, params);

      // Wall-clock budget and UI progress knobs for the drive loop
      long startedAtNanos = System.nanoTime();
      long deadlineNanos = this.resolveDeadlineNanos(timeout, startedAtNanos);
      boolean showProgress = this.shouldShowProgress(useTqdm);
      Map<Integer, List<Integer>> outputsBySeqId = new HashMap<>();

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
      return this.decodeCompletedOutputs(outputsBySeqId);
    }
  }

  private void beginGeneration() {
    // Internal: this generate owns the cancel flag until finishGeneration
    this.cancelRequested.set(false);
  }

  private void finishGeneration() {
    // Internal: incomplete work after abort must free KV pages via scheduler.clear()
    if (!this.isFinished()) {
      this.scheduler.clear();
    }
    this.cancelRequested.set(false);
  }

  private void enqueueAllPrompts(List<?> prompts, List<SamplingParams> params) {
    // Internal: one Sequence per prompt, paired with its SamplingParams
    IntStream.range(0, prompts.size())
        .forEach(i -> this.enqueuePrompt(prompts.get(i), params.get(i)));
  }

  private void enqueuePrompt(Object prompt, SamplingParams samplingParams) {
    // Internal: accept either raw text (tokenize) or pre-tokenized ids
    if (prompt instanceof String text) {
      this.addRequest(text, samplingParams);
      return;
    }
    if (prompt instanceof List<?> ids) {
      this.addRequest(this.toTokenIdList(ids), samplingParams);
      return;
    }
    throw new IllegalArgumentException("prompt must be String or List<Integer>");
  }

  private List<Integer> toTokenIdList(List<?> ids) {
    // Internal: untyped List<?> from the public generate API → List<Integer>
    return ids.stream()
        .map(id -> ((Number) id).intValue())
        .toList();
  }

  private void driveUntilSchedulerIdle(
      Duration timeout,
      long deadlineNanos,
      java.util.function.IntConsumer onToken,
      boolean showProgress,
      int totalPrompts,
      long startedAtNanos,
      Map<Integer, List<Integer>> outputsBySeqId
  ) {
    // Business loop: keep producing tokens until every sequence is finished or aborted
    int completed = 0;
    while (!this.isFinished()) {
      // Guard: cancel flag or wall-clock deadline → throw (finally clears the scheduler)
      this.requireGenerationAllowed(timeout, deadlineNanos);

      // One engine tick (schedule → forward+sample → postprocess)
      StepResult step = this.step();

      // Optional streaming: notify caller for each newly appended token id
      this.dispatchTokenEvents(step, onToken);

      // Collect finished sequences; update progress counter when a prompt completes
      completed = this.recordFinishedOutputs(
          step, outputsBySeqId, completed, showProgress, totalPrompts, startedAtNanos);
    }
  }

  private void requireGenerationAllowed(Duration timeout, long deadlineNanos) {
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

  private void requireWithinDeadline(Duration timeout, long deadlineNanos) {
    // Internal: Long.MAX_VALUE means “no timeout”
    if (System.nanoTime() > deadlineNanos) {
      throw new GenerationTimeoutException(timeout);
    }
  }

  private void dispatchTokenEvents(StepResult step, java.util.function.IntConsumer onToken) {
    // Internal: skip when the caller did not request streaming
    if (onToken == null) {
      return;
    }
    step.tokenEvents().stream()
        .mapToInt(TokenEvent::tokenId)
        .forEach(onToken);
  }

  private int recordFinishedOutputs(
      StepResult step,
      Map<Integer, List<Integer>> outputsBySeqId,
      int completed,
      boolean showProgress,
      int totalPrompts,
      long startedAtNanos
  ) {
    // Internal: stash completion token ids by seqId; progress counts finished prompts
    int nextCompleted = completed;
    for (FinishedOutput finished : step.outputs()) {
      outputsBySeqId.put(finished.seqId(), finished.tokenIds());
      nextCompleted++;
      this.reportProgressIfNeeded(showProgress, nextCompleted, totalPrompts, startedAtNanos);
    }
    return nextCompleted;
  }

  private void reportProgressIfNeeded(
      boolean showProgress,
      int completed,
      int totalPrompts,
      long startedAtNanos
  ) {
    // Internal: CLI-style progress only when requested and EngineIo is not silent
    if (!showProgress) {
      return;
    }
    double elapsedSeconds = (System.nanoTime() - startedAtNanos) / 1e9;
    this.io.progressf("\rGenerating %d/%d (%.1fs)", completed, totalPrompts, elapsedSeconds);
  }

  private void finishProgressLine(boolean showProgress) {
    // Internal: move past the \r progress line before returning text
    if (showProgress) {
      this.io.out().println();
    }
  }

  private long resolveDeadlineNanos(Duration timeout, long startedAtNanos) {
    // Internal: null / zero / negative Duration → unbounded generate
    if (timeout == null || timeout.isZero() || timeout.isNegative()) {
      return Long.MAX_VALUE;
    }
    return startedAtNanos + timeout.toNanos();
  }

  private boolean shouldShowProgress(boolean useTqdm) {
    // Internal: never spam progress into silent (library-default) I/O
    return useTqdm && !this.io.isSilent();
  }

  private List<GenerationOutput> decodeCompletedOutputs(
      Map<Integer, List<Integer>> outputsBySeqId) {
    // Internal: stable seqId order → GenerationOutput(text, tokenIds) for each prompt
    return outputsBySeqId.keySet().stream()
        .sorted()
        .map(seqId -> {
          List<Integer> tokenIds = outputsBySeqId.get(seqId);
          return new GenerationOutput(this.tokenizer.decode(tokenIds, true), tokenIds);
        })
        .toList();
  }

  private List<SamplingParams> resolveSamplingParams(List<?> prompts, Object samplingParams) {
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
   * Model-aware defaults (e.g. Gemma {@code top_k=64}) with {@link SamplingDefaults#DEFAULT_MAX_TOKENS}.
   */
  public SamplingParams defaultSampling() {
    return SamplingDefaults.forTokenizer(this.tokenizer);
  }

  /**
   * Model-aware defaults with a custom max new-token budget.
   */
  public SamplingParams defaultSampling(int maxTokens) {
    return SamplingDefaults.forTokenizer(this.tokenizer, maxTokens);
  }

  /**
   * Opens a multi-turn {@link ChatSession} using {@link #defaultSampling()} and this instance's
   * {@link #systemPrompt()}.
   */
  public ChatSession chat() {
    return new ChatSession(this);
  }

  /**
   * Opens a multi-turn session with explicit sampling.
   */
  public ChatSession chat(SamplingParams samplingParams) {
    return new ChatSession(this, samplingParams);
  }

  /**
   * Opens a multi-turn session with model-aware sampling limited to {@code maxTokens}.
   */
  public ChatSession chat(int maxTokens) {
    return ChatSession.open(this, maxTokens);
  }

  /**
   * Opens a retrieval-augmented session over {@code index}
   * (a {@link PreparedRag} from {@link RagFactory}, or any {@link RagIndex}).
   */
  public RagSession rag(RagIndex index) {
    return RagSession.open(this, index);
  }

  /**
   * Opens a RAG session with model-aware sampling limited to {@code maxTokens}.
   */
  public RagSession rag(RagIndex index, int maxTokens) {
    return RagSession.open(this, index, maxTokens);
  }

  /**
   * Raw text completion (no chat template). Uses {@link #defaultSampling()}.
   *
   * @return decoded completion text for the single prompt
   */
  public String complete(String prompt) {
    return this.complete(prompt, this.defaultSampling());
  }

  /**
   * Raw text completion (no chat template).
   *
   * @param prompt         continuation seed as plain text
   * @param samplingParams sampling controls
   * @return decoded completion text
   */
  public String complete(String prompt, SamplingParams samplingParams) {
    // Business: raw continuation (no chat template) → single decoded string
    requireNonNull(prompt, "prompt");
    List<GenerationOutput> outputs = this.generate(List.of(prompt), samplingParams);
    return outputs.getFirst().text();
  }

  /**
   * Single-turn chat: system prompt (if any) + one user message, then the assistant answer text.
   * Uses {@link #defaultSampling()}.
   */
  public String chatOnce(String userMessage) {
    return this.chatOnce(userMessage, this.defaultSampling());
  }

  /**
   * Single-turn chat with explicit sampling. History is not retained after the call.
   */
  public String chatOnce(String userMessage, SamplingParams samplingParams) {
    // Business: one ChatSession turn; return visible answer only (thinking stripped by session)
    ChatReply reply = this.chat(samplingParams).send(userMessage);
    return reply.answer();
  }

  public Tokenizer tokenizer() {
    return this.tokenizer;
  }

  public Config config() {
    return this.config;
  }

  /**
   * Status / progress streams used by load and optional generate progress.
   */
  public EngineIo io() {
    return this.io;
  }

  /**
   * System text used by {@link #newConversation()} and {@link #chat()}.
   * Builder override wins; otherwise {@link ChatPrompts#systemFor(boolean)}
   * (empty for Gemma by default).
   */
  public String systemPrompt() {
    if (this.systemPromptOverride != null) {
      return this.systemPromptOverride;
    }
    return ChatPrompts.systemFor(this.tokenizer.isGemmaChat());
  }

  /**
   * Fresh dialog history: optional system turn from {@link #systemPrompt()}, nothing else.
   */
  public List<ChatMessage> newConversation() {
    return ChatMessages.newConversation(this.systemPrompt());
  }

  /**
   * Cancels any in-flight generate and releases runner resources.
   */
  @Override
  public void close() {
    // Business: stop in-flight generate, then release runner / Context under the generate lock
    this.cancel();
    synchronized (this.generateLock) {
      this.modelRunner.close();
    }
  }

  /**
   * Fluent configurator for {@link LLM}.
   *
   * <p>Defaults: {@link EngineIo#silent()}, eager execution, warmup enabled, no system-prompt
   * override (model default via {@link ChatPrompts}).
   *
   * <p>Provide either a shared {@link Model} via {@link LLM#builder(Model)} or a model directory
   * via {@link LLM#builder(Path)}. Call {@link #build()} last.
   *
   * @throws ModelLoadException from {@link #build()} when weights or config cannot be loaded
   */
  public static final class Builder {

    private final Model sharedModel;
    private final Path modelDir;
    private EngineIo io = EngineIo.silent();
    private int maxNumBatchedTokens = 16384;
    private int maxNumSeqs = 512;
    private int maxModelLen = 4096;
    private float gpuMemoryUtilization = 0.9f;
    private int tensorParallelSize = 1;
    private boolean enforceEager = true;
    private int kvcacheBlockSize = 256;
    private int numKvcacheBlocks = -1;
    private boolean warmup = true;
    /**
     * {@code null} = model default from {@link ChatPrompts}; blank = no system turn.
     */
    private String systemPromptOverride = null;

    private Builder(Model model) {
      this.sharedModel = requireNonNull(model, "model");
      this.modelDir = model.path();
    }

    private Builder(Path modelDir) {
      this.sharedModel = null;
      this.modelDir = requireNonNull(modelDir, "model");
    }

    /**
     * Custom status streams; {@code null} is treated as {@link EngineIo#silent()}.
     */
    public Builder io(EngineIo io) {
      this.io = io == null ? EngineIo.silent() : io;
      return this;
    }

    /**
     * Routes engine status to {@link System#out} / {@link System#err} (CLI-friendly).
     */
    public Builder withSystemIo() {
      return this.io(EngineIo.system());
    }

    /**
     * Explicit quiet mode (also the default).
     */
    public Builder quiet() {
      return this.io(EngineIo.silent());
    }

    /**
     * Sets the chat system prompt. {@code null} becomes blank (no system turn).
     * Omit this method entirely to keep the model default from {@link ChatPrompts}.
     */
    public Builder systemPrompt(String prompt) {
      this.systemPromptOverride = prompt == null ? "" : prompt;
      return this;
    }

    /**
     * Forces no system turn in {@link LLM#newConversation()} / chat helpers.
     */
    public Builder noSystemPrompt() {
      return this.systemPrompt("");
    }

    /**
     * Clears an override so {@link ChatPrompts#systemFor(boolean)} applies again.
     */
    public Builder defaultSystemPrompt() {
      this.systemPromptOverride = null;
      return this;
    }

    /**
     * Max tokens across a prefill batch. Default {@code 16384}.
     */
    public Builder maxNumBatchedTokens(int value) {
      this.maxNumBatchedTokens = value;
      return this;
    }

    /**
     * Max concurrent sequences in the scheduler. Default {@code 512}.
     */
    public Builder maxNumSeqs(int value) {
      this.maxNumSeqs = value;
      return this;
    }

    /**
     * Max context length in tokens (capped by the model's {@code max_position_embeddings}).
     * Default {@code 4096}.
     */
    public Builder maxModelLen(int value) {
      this.maxModelLen = value;
      return this;
    }

    /**
     * Fraction of heap used when estimating KV-cache size. Default {@code 0.9}.
     * (Named for upstream parity; this port is CPU/heap-based.)
     */
    public Builder gpuMemoryUtilization(float value) {
      this.gpuMemoryUtilization = value;
      return this;
    }

    /**
     * Currently only {@code 1} is supported.
     */
    public Builder tensorParallelSize(int value) {
      this.tensorParallelSize = value;
      return this;
    }

    /**
     * When {@code true} (default), runs eager forward passes only (no CUDA-graph style capture).
     * Required {@code true} on this CPU port.
     */
    public Builder enforceEager(boolean value) {
      this.enforceEager = value;
      return this;
    }

    /**
     * KV block size in tokens; must be a multiple of 256. Default {@code 256}.
     */
    public Builder kvcacheBlockSize(int value) {
      this.kvcacheBlockSize = value;
      return this;
    }

    /**
     * Number of KV blocks to allocate. {@code -1} (default) estimates from heap and config.
     */
    public Builder numKvcacheBlocks(int value) {
      this.numKvcacheBlocks = value;
      return this;
    }

    /**
     * When {@code true} (default), runs a short generate after load to warm JIT / caches.
     */
    public Builder warmup(boolean value) {
      this.warmup = value;
      return this;
    }

    /**
     * Skips post-load warmup (faster construct; first real generate pays JIT cost).
     */
    public Builder skipWarmup() {
      return this.warmup(false);
    }

    /**
     * Returns a ready {@link LLM} bound to the resolved {@link Model}.
     *
     * @throws ModelLoadException if the model directory, config, or weights are unusable
     */
    public LLM build() {
      return new LLM(this);
    }

    private Model resolveModel() {
      if (this.sharedModel != null) {
        return this.sharedModel;
      }
      return ModelFactory.make(this.modelDir, this.io);
    }

    private Path modelPath() {
      return this.sharedModel != null ? this.sharedModel.path() : this.modelDir;
    }

    private Config toConfig(Model model) {
      return Config.builder(model)
          .maxNumBatchedTokens(this.maxNumBatchedTokens)
          .maxNumSeqs(this.maxNumSeqs)
          .maxModelLen(this.maxModelLen)
          .gpuMemoryUtilization(this.gpuMemoryUtilization)
          .tensorParallelSize(this.tensorParallelSize)
          .enforceEager(this.enforceEager)
          .kvcacheBlockSize(this.kvcacheBlockSize)
          .numKvcacheBlocks(this.numKvcacheBlocks)
          .build();
    }
  }

  /** A finished sequence's completion token ids. */
  public record FinishedOutput(int seqId, List<Integer> tokenIds) {
  }

  /** One decoded token belonging to {@code seqId}. */
  public record TokenEvent(int seqId, int tokenId) {
  }

  /** Result of a single {@link LLM#step()}. */
  public record StepResult(List<FinishedOutput> outputs, List<TokenEvent> tokenEvents,
                           int numTokens) {
    public StepResult(List<FinishedOutput> outputs, int numTokens) {
      this(outputs, List.of(), numTokens);
    }
  }

  /** Decoded text and token ids for one completed prompt in {@link LLM#generate}. */
  public record GenerationOutput(String text, List<Integer> tokenIds) {
  }
}
