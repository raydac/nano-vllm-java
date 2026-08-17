package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Tag("concurrent_test")
class ConcurrentLibraryUseTest {

  private static final List<Station> STATIONS = List.of(
    new Station("Alpha", "NVJZX-ALPHA-17"),
    new Station("Bravo", "NVJZX-BRAVO-29"),
    new Station("Charlie", "NVJZX-CHARLIE-31"),
    new Station("Delta", "NVJZX-DELTA-47"),
    new Station("Echo", "NVJZX-ECHO-53"),
    new Station("Foxtrot", "NVJZX-FOXTROT-67"),
    new Station("Hotel", "NVJZX-HOTEL-71"),
    new Station("India", "NVJZX-INDIA-83"),
    new Station("Juliet", "NVJZX-JULIET-89"),
    new Station("Kilo", "NVJZX-KILO-97"),
    new Station("Lima", "NVJZX-LIMA-101"),
    new Station("Mike", "NVJZX-MIKE-103"));
  private static final int WORKERS = 6;
  private static final int HAMMERS = 8;
  private static final int ROUNDS = 3;
  private static final int TURNS_PER_ROUND = 3;
  private static final int FILLER_DOCS = 80;
  private static final int RETRIEVE_TOP_K = 8;
  private static final int SESSION_TOP_K = 4;
  private static final int RETRIEVE_BURST = 8;
  private static final int MAX_NEW_TOKENS = 32;
  private static final Duration SEND_TIMEOUT = Duration.ofMinutes(2);
  private static final Duration GATE_TIMEOUT = Duration.ofMinutes(3);
  private static final Duration JOIN_TIMEOUT = Duration.ofMinutes(8);
  private static final String SHARED_FACT = "Paris is the capital of France.";
  private static final String SHARED_QUESTION = "What city is the capital of France?";
  private static final String SHARED_MARKER = "Paris";

  private final AtomicInteger liveWorkers = new AtomicInteger();
  private final AtomicInteger liveHammers = new AtomicInteger();
  private final AtomicInteger liveTurns = new AtomicInteger();
  private long testStartedNanos;

  private static ThreadFactory namedWorkers() {
    AtomicInteger seq = new AtomicInteger();
    return runnable -> {
      Thread thread = new Thread(runnable, "nanollvm-concurrent-test-" + seq.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    };
  }

  @Test
  @Timeout(value = 8, unit = TimeUnit.MINUTES)
  void sharedModelAndPreparedRagSurviveRacedSessions() {
    this.testStartedNanos = System.nanoTime();
    this.log(
      "plan workers=%d hammers=%d rounds=%d turns/worker=%d maxTokens=%d"
        .formatted(WORKERS, HAMMERS, ROUNDS, ROUNDS * TURNS_PER_ROUND, MAX_NEW_TOKENS));

    Path chatPath = OptionalModelAssumptions.requireSmolLm2InstructOnnx();
    Path embedPath = OptionalModelAssumptions.requireGteSmallGguf();
    this.log("loading chat=%s embed=%s".formatted(chatPath.getFileName(), embedPath.getFileName()));
    PreparedRag lexical = this.prepareCorpus();
    this.log("lexical corpus chunks=%d".formatted(lexical.size()));
    this.assertSequentialRetrieval(lexical);

    CyclicBarrier start = new CyclicBarrier(WORKERS + HAMMERS);
    AtomicBoolean keepRetrieving = new AtomicBoolean(true);
    AtomicInteger retrieveOps = new AtomicInteger();

    try (LlmModel embedModel = LlmModelFactory.make(embedPath);
         LlmModel chatModel = LlmModelFactory.make(chatPath);
         ExecutorService pool = Executors.newFixedThreadPool(
           WORKERS + HAMMERS, namedWorkers())) {
      this.log("models loaded, building hybrid index");
      RagIndex index = RagFactory.withEmbeddings(lexical, embedModel);
      this.assertSequentialRetrieval(index);
      this.log(
        "hybrid ready, submitting %d generate workers + %d retrieve hammers"
          .formatted(WORKERS, HAMMERS));

      List<Future<WorkerReport>> workers = IntStream.range(0, WORKERS)
        .mapToObj(worker -> pool.submit(
          () -> this.runWorker(worker, chatModel, lexical, index, start, keepRetrieving)))
        .toList();
      List<Future<Integer>> hammers = IntStream.range(0, HAMMERS)
        .mapToObj(hammer -> pool.submit(
          () -> this.runHammer(
            hammer,
            hammer % 2 == 0 ? index : lexical,
            start,
            keepRetrieving,
            retrieveOps)))
        .toList();

      List<WorkerReport> reports = this.awaitAll(workers);
      keepRetrieving.set(false);
      int hammerRetrieves = this.awaitHammers(hammers);

      assertEquals(WORKERS, reports.size());
      reports.forEach(this::assertWorkerFinishedCleanly);
      assertTrue(
        hammerRetrieves >= WORKERS * RETRIEVE_BURST,
        () -> "retrieve hammers did too little work: " + hammerRetrieves);
      assertTrue(retrieveOps.get() >= hammerRetrieves);
      this.log(
        "done workers=%d hammers=%d retrieveOps=%d"
          .formatted(reports.size(), HAMMERS, retrieveOps.get()));
    }
  }

  private PreparedRag prepareCorpus() {
    RagFactory.Builder builder = RagFactory.builder()
      .forTinyModels()
      .add("paris.txt", SHARED_FACT);
    STATIONS.forEach(station -> builder.add(station.sourceId(), station.document()));
    IntStream.range(0, FILLER_DOCS).forEach(aisle -> builder.add(
      "aisle-%d.txt".formatted(aisle),
      "Aisle %d holds pallets of dry goods, rope, and spare fittings in the general shed."
        .formatted(aisle)));
    PreparedRag corpus = builder.build();
    assertTrue(corpus.size() >= 1 + STATIONS.size() + FILLER_DOCS);
    return corpus;
  }

  private void assertSequentialRetrieval(final RagIndex corpus) {
    this.assertHitsContain(corpus.retrieve(SHARED_QUESTION, RETRIEVE_TOP_K), SHARED_MARKER);
    STATIONS.forEach(station ->
      this.assertHitsContain(corpus.retrieve(station.question(), RETRIEVE_TOP_K),
        station.codeword()));
  }

  private WorkerReport runWorker(
    final int index,
    final LlmModel model,
    final PreparedRag lexical,
    final RagIndex hybrid,
    final CyclicBarrier start,
    final AtomicBoolean keepRetrieving
  ) {
    Station station = STATIONS.get(index);
    StringBuilder raw = new StringBuilder();
    List<String> debugPrompts = new ArrayList<>();
    this.liveWorkers.incrementAndGet();
    this.log("start worker %s engine".formatted(station.name()));

    try (LLM llm = this.openEngine(model)) {
      assertEquals(model, llm.model());
      RagSession rag = this.openRag(llm, hybrid, raw, debugPrompts);
      this.log("worker %s waiting for race gate".formatted(station.name()));
      this.awaitGate(start);
      this.log("worker %s racing".formatted(station.name()));

      List<TurnReport> turns = new ArrayList<>();
      for (int round = 0; round < ROUNDS; round++) {
        this.contendIndexAndTokenizer(model, lexical, hybrid, station);
        turns.add(this.askAndCheck(
          rag, station.question(), station.codeword(), station, raw, debugPrompts, round));

        this.contendIndexAndTokenizer(model, lexical, hybrid, station);
        turns.add(this.askAndCheck(
          rag, station.followUp(), station.codeword(), station, raw, debugPrompts, round));

        this.contendIndexAndTokenizer(model, lexical, hybrid, station);
        turns.add(this.askAndCheck(
          rag, SHARED_QUESTION, SHARED_MARKER, station, raw, debugPrompts, round));
      }

      this.assertHistoryStaysOnThisWorker(rag, station);
      return new WorkerReport(station, List.copyOf(turns));
    } catch (AssertionError | RuntimeException error) {
      keepRetrieving.set(false);
      this.failGates(start);
      throw error;
    } finally {
      this.liveWorkers.decrementAndGet();
      this.log("end worker %s".formatted(station.name()));
    }
  }

  private int runHammer(
    final int index,
    final RagIndex corpus,
    final CyclicBarrier start,
    final AtomicBoolean keepRetrieving,
    final AtomicInteger retrieveOps
  ) {
    Station station = STATIONS.get(index % STATIONS.size());
    String indexKind = corpus instanceof PreparedRag ? "lexical" : "hybrid";
    int localOps = 0;
    this.liveHammers.incrementAndGet();
    this.log("start hammer %d %s".formatted(index, indexKind));
    try {
      this.log("hammer %d waiting for race gate".formatted(index));
      this.awaitGate(start);
      this.log("hammer %d racing".formatted(index));
      while (keepRetrieving.get()) {
        this.assertHitsContain(corpus.retrieve(SHARED_QUESTION, RETRIEVE_TOP_K), SHARED_MARKER);
        this.assertHitsContain(corpus.retrieve(station.question(), RETRIEVE_TOP_K),
          station.codeword());
        corpus.retrieve("aisle pallets dry goods rope", RETRIEVE_TOP_K);
        localOps += 3;
        retrieveOps.addAndGet(3);
      }
      return localOps;
    } catch (AssertionError | RuntimeException error) {
      keepRetrieving.set(false);
      this.failGates(start);
      throw error;
    } finally {
      this.liveHammers.decrementAndGet();
      this.log("end hammer %d retrieves=%d".formatted(index, localOps));
    }
  }

  private void contendIndexAndTokenizer(
    final LlmModel model,
    final PreparedRag lexical,
    final RagIndex hybrid,
    final Station station
  ) {
    Tokenizer tokenizer = model.tokenizer();
    IntStream.range(0, RETRIEVE_BURST).forEach(step -> {
      String query = switch (step % 3) {
        case 0 -> SHARED_QUESTION;
        case 1 -> station.question();
        default -> "aisle %d pallets dry goods".formatted(step);
      };
      RagIndex target = step % 2 == 0 ? hybrid : lexical;
      List<RagHit> hits = target.retrieve(query, RETRIEVE_TOP_K);
      if (query.equals(SHARED_QUESTION)) {
        this.assertHitsContain(hits, SHARED_MARKER);
      } else if (query.equals(station.question())) {
        this.assertHitsContain(hits, station.codeword());
      }
      List<Integer> ids = tokenizer.encode(query);
      assertFalse(ids.isEmpty());
      assertFalse(tokenizer.decode(ids, true).isBlank());
    });
  }

  private LLM openEngine(final LlmModel model) {
    return LLM.builder(model)
      .maxModelLen(512)
      .numKvcacheBlocks(32)
      .kvHeapFraction(0.2f)
      .cpuThreads(2)
      .sampling(SamplingParams.builder()
        .temperature(0.15f)
        .maxTokens(MAX_NEW_TOKENS)
        .topP(0.9f)
        .build())
      .systemPrompt(
        "Answer using only the provided facts. Reply with the inventory code or the city name.")
      .build();
  }

  private RagSession openRag(
    final LLM llm,
    final RagIndex corpus,
    final StringBuilder raw,
    final List<String> debugPrompts
  ) {
    return llm.rag(corpus, MAX_NEW_TOKENS)
      .topK(SESSION_TOP_K)
      .maxContextChars(1800)
      .maxHistoryMessages(64)
      .isolateGeneration(true)
      .enableThinking(false)
      .timeout(SEND_TIMEOUT)
      .emitDebugPrompts(true)
      .listen(this.capturing(raw, debugPrompts));
  }

  private LlmListener capturing(final StringBuilder raw, final List<String> debugPrompts) {
    return (source, event) -> {
      switch (event.kind()) {
        case TEXT_RAW -> {
          if (event.snapshot()) {
            raw.setLength(0);
          }
          raw.append(event.text());
        }
        case TEXT_DEBUG -> debugPrompts.add(event.text());
        default -> {
        }
      }
    };
  }

  private TurnReport askAndCheck(
    final RagSession rag,
    final String question,
    final String expectedMarker,
    final Station station,
    final StringBuilder raw,
    final List<String> debugPrompts,
    final int round
  ) {
    raw.setLength(0);
    debugPrompts.clear();
    this.liveTurns.incrementAndGet();
    this.log("start turn %s round %d/%d q=%s".formatted(
      station.name(), round + 1, ROUNDS, question));
    long turnStartedNanos = System.nanoTime();

    try {
      ChatReply reply = rag.send(question);
      String generated = this.generatedText(reply, raw);

      this.assertHitsContain(rag.lastHits(), expectedMarker);

      assertFalse(reply.thinkOpen(),
        () -> this.turnFailure(station, question, reply, generated, rag));
      assertTrue(
        reply.stats().promptTokens() > 0,
        () -> this.turnFailure(station, question, reply, generated, rag));
      assertTrue(
        reply.stats().completionTokens() > 0,
        () -> this.turnFailure(station, question, reply, generated, rag));
      assertTrue(
        generated.chars().anyMatch(Character::isLetter),
        () -> this.turnFailure(station, question, reply, generated, rag));
      assertTrue(
        debugPrompts.stream().anyMatch(prompt -> this.containsIgnoreCase(prompt, expectedMarker)),
        () -> "worker %s prompt missed %s: debug=%s generated=%s".formatted(
          station.name(), expectedMarker, debugPrompts, generated));

      this.log(String.format(Locale.ROOT, "end turn %s round %d/%d tokens=%d+%d %.1fs",
        station.name(),
        round + 1,
        ROUNDS,
        reply.stats().promptTokens(),
        reply.stats().completionTokens(),
        (System.nanoTime() - turnStartedNanos) / 1e9));

      return new TurnReport(
        question, expectedMarker, generated, reply.stats().promptTokens(),
        reply.stats().completionTokens());
    } finally {
      this.liveTurns.decrementAndGet();
    }
  }

  private void assertHistoryStaysOnThisWorker(final RagSession rag, final Station station) {
    List<String> userTurns = rag.chat().history().stream()
      .filter(message -> message.role() == ChatRole.USER)
      .map(ChatMessage::content)
      .toList();
    assertEquals(ROUNDS * TURNS_PER_ROUND, userTurns.size(), () -> "history=" + userTurns);
    assertTrue(
      userTurns.stream().allMatch(turn ->
        turn.contains(station.name()) || turn.contains("capital of France")),
      () -> "worker %s mixed questions: %s".formatted(station.name(), userTurns));

    List<String> foreignNames = STATIONS.stream()
      .map(Station::name)
      .filter(name -> !name.equals(station.name()))
      .toList();
    assertTrue(
      userTurns.stream().noneMatch(turn -> foreignNames.stream().anyMatch(turn::contains)),
      () -> "worker %s saw foreign depots: %s".formatted(station.name(), userTurns));
  }

  private void assertWorkerFinishedCleanly(final WorkerReport report) {
    assertEquals(ROUNDS * TURNS_PER_ROUND, report.turns().size(), report.station().name());
    report.turns().forEach(turn -> {
      assertFalse(turn.generated().isBlank(), report.station().name() + ": " + turn.question());
      assertTrue(turn.promptTokens() > 0, turn.question());
      assertTrue(turn.completionTokens() > 0, turn.question());
    });
  }

  private void assertHitsContain(final List<RagHit> hits, final String marker) {
    assertFalse(hits.isEmpty(), () -> "no hits for marker " + marker);
    assertTrue(
      hits.stream().anyMatch(hit -> this.containsIgnoreCase(hit.chunk().text(), marker)),
      () -> "expected %s in %s".formatted(
        marker,
        hits.stream().map(hit -> hit.chunk().text()).toList()));
  }

  private String generatedText(final ChatReply reply, final StringBuilder raw) {
    return (reply.answer() + " " + reply.thinking() + " " + raw).strip();
  }

  private boolean containsIgnoreCase(final String text, final String marker) {
    return text.toLowerCase(Locale.ROOT).contains(marker.toLowerCase(Locale.ROOT));
  }

  private String turnFailure(
    final Station station,
    final String question,
    final ChatReply reply,
    final String generated,
    final RagSession rag
  ) {
    return "worker %s q=%s answer=%s thinking=%s generated=%s stats=%s hits=%s".formatted(
      station.name(),
      question,
      reply.answer(),
      reply.thinking(),
      generated,
      reply.stats(),
      rag.lastHits().stream().map(hit -> hit.chunk().text()).toList());
  }

  private List<WorkerReport> awaitAll(final List<Future<WorkerReport>> futures) {
    List<WorkerReport> reports = new ArrayList<>();
    List<Throwable> errors = new ArrayList<>();
    for (Future<WorkerReport> future : futures) {
      try {
        reports.add(future.get(JOIN_TIMEOUT.toSeconds(), TimeUnit.SECONDS));
      } catch (TimeoutException e) {
        future.cancel(true);
        errors.add(new AssertionError("worker did not finish in " + JOIN_TIMEOUT, e));
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        errors.add(new AssertionError("interrupted waiting for worker", e));
      } catch (ExecutionException e) {
        errors.add(e.getCause() == null ? e : e.getCause());
      }
    }
    if (!errors.isEmpty()) {
      AssertionError failed = new AssertionError(
        errors.size() + " workers failed: "
          + errors.stream().map(Throwable::toString).toList());
      errors.forEach(failed::addSuppressed);
      throw failed;
    }
    return reports;
  }

  private int awaitHammers(final List<Future<Integer>> hammers) {
    List<Throwable> errors = new ArrayList<>();
    int total = 0;
    for (Future<Integer> hammer : hammers) {
      try {
        total += hammer.get(JOIN_TIMEOUT.toSeconds(), TimeUnit.SECONDS);
      } catch (TimeoutException e) {
        hammer.cancel(true);
        errors.add(new AssertionError("hammer did not finish in " + JOIN_TIMEOUT, e));
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        errors.add(new AssertionError("interrupted waiting for hammer", e));
      } catch (ExecutionException e) {
        errors.add(e.getCause() == null ? e : e.getCause());
      }
    }
    if (!errors.isEmpty()) {
      AssertionError failed = new AssertionError(
        errors.size() + " hammers failed: "
          + errors.stream().map(Throwable::toString).toList());
      errors.forEach(failed::addSuppressed);
      throw failed;
    }
    return total;
  }

  private void awaitGate(final CyclicBarrier gate) {
    try {
      gate.await(GATE_TIMEOUT.toSeconds(), TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new AssertionError("interrupted at race gate", e);
    } catch (BrokenBarrierException | TimeoutException e) {
      throw new AssertionError("race gate failed", e);
    }
  }

  private void failGates(final CyclicBarrier... gates) {
    for (CyclicBarrier gate : gates) {
      gate.reset();
    }
  }

  private void log(final String message) {
    double elapsed = this.testStartedNanos == 0L
      ? 0d
      : (System.nanoTime() - this.testStartedNanos) / 1e9;
    synchronized (System.err) {
      System.err.printf(Locale.ROOT, "[nanollvm-concurrent] +%6.1fs  %s  [%s]%n",
        elapsed, message, this.parallelSnapshot());
    }
  }

  private String parallelSnapshot() {
    return "parallel workers=%d/%d hammers=%d/%d turns=%d".formatted(
      this.liveWorkers.get(),
      WORKERS,
      this.liveHammers.get(),
      HAMMERS,
      this.liveTurns.get());
  }

  private record Station(String name, String codeword) {
    private String sourceId() {
      return this.name.toLowerCase(Locale.ROOT) + ".txt";
    }

    private String document() {
      return """
        Depot %s is a unique warehouse in sector %s. It handles sealed crates and bonded cargo.
        The only inventory code at depot %s is %s.
        Inspectors must quote %s when opening depot %s lockers.
        """.formatted(
        this.name, this.name, this.name, this.codeword, this.codeword, this.name);
    }

    private String question() {
      return "What is the only inventory code at depot %s?".formatted(this.name);
    }

    private String followUp() {
      return "Repeat the inventory code for depot %s.".formatted(this.name);
    }
  }

  private record TurnReport(
    String question,
    String expectedMarker,
    String generated,
    int promptTokens,
    int completionTokens
  ) {
  }

  private record WorkerReport(Station station, List<TurnReport> turns) {
  }
}
