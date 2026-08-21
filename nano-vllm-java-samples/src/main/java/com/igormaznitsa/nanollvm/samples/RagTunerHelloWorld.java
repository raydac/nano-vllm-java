package com.igormaznitsa.nanollvm.samples;

import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.DenseRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.rag.RagTuner;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.EpubText;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Dense embedding RAG over a bundled EPUB: {@link RagTuner} filter / extract / preprocess,
 * JDK zip + StAX for the book text, then cosine retrieval with an embedding {@link LlmModel}.
 *
 * <p>Indexes Project Gutenberg {@code pg59112.epub} (Karel Čapek, <em>R.U.R.</em>) from the
 * samples classpath, unpacks gte-small and embeds every chunk (optional caller {@code Executor}
 * for parallel forwards, with an in-place percent/ETA bar), then asks questions that are answered
 * in that play.
 *
 * <p>Args: optional chat model directory (default {@code models/Qwen3-0.6B}), optional embedding
 * checkpoint (default {@code models/gte-small.Q2_K.gguf}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.RagTunerHelloWorld}
 *
 * @since 1.1.1
 */
public final class RagTunerHelloWorld {

  private RagTunerHelloWorld() {
  }

  public static void main(final String[] args) {
    Path chatDir = chatModel(args);
    Path embedPath = embeddingModel(args);

    System.out.println("Loading embedding model from " + embedPath);
    System.out.println("EPUB corpus classpath:" + EpubText.RUR_EPUB);

    long started = System.currentTimeMillis();
    PreparedRag documents = loadBundledEpub();
    System.out.println("Chunks to embed: " + documents.size());

    int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
    try (LlmModel embed = LlmModelFactory.open(embedPath).unpackParameters().make();
         ExecutorService embedPool = Executors.newFixedThreadPool(workers)) {
      int total = documents.size();
      System.out.println(
        "Embedding " + total + " passages with " + embed.architectureName()
          + " (unpacked GGUF, " + workers + " embed threads)…");
      EmbedProgress progress = new EmbedProgress(total);
      DenseRagIndex index = DenseRagIndex.of(documents, embed, embedPool, progress::embedded);
      progress.finish();
      System.out.println("Dense index: " + index.size() + " vectors, dim=" + index.dimensions());

      System.out.println("Loading chat model from " + chatDir);
      try (LlmModel chat = LlmModelFactory.make(chatDir);
           LLM llm = LLM.builder(chat)
             .noSystemPrompt()
             .maxModelLen(2048)
             .build()) {
        RagSession rag = llm.rag(index, 128)
          .topK(3)
          .maxContextChars(1800)
          .isolateGeneration(llm.tokenizer().isTurnBasedChat())
          .sampling(SampleChatPrompts.samplingForDemo(llm.tokenizer(), 128)
            .withTemperature(0.1f)
            .withTopP(0.8f))
          .recoverUnusableAnswers(true)
          .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
          .streamTo(System.err, System.out, false);

        ask(rag, "What does the Czech word robot mean?");
        ask(rag, "Who is Helena Glory?");
      }
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  static PreparedRag loadBundledEpub() {
    return RagFactory.builder()
      .options(RagLoadOptions.defaults().withMaxChunkChars(128).withChunkOverlap(0))
      .addProcessor(
        RagTuner.allowing(EpubText::isEpub),
        RagTuner.extracting(EpubText::extract),
        RagTuner.preprocessing(EpubText::normalizeWhitespace))
      .addResource(RagTunerHelloWorld.class, "/" + EpubText.RUR_EPUB)
      .build();
  }

  private static Path chatModel(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return Path.of(args[0]).toAbsolutePath().normalize();
    }
    return BundledModels.require(BundledModels.QWEN3_0_6B);
  }

  private static Path embeddingModel(final String[] args) {
    if (args != null && args.length > 1 && args[1] != null && !args[1].isBlank()) {
      return Path.of(args[1]).toAbsolutePath().normalize();
    }
    return BundledModels.require(BundledModels.GTE_SMALL_GGUF);
  }

  private static void ask(final RagSession rag, final String question) {
    System.out.println();
    System.out.println("Q: " + question);
    rag.send(question);
    System.out.println();
    sleep(500);
    printHits(rag.lastHits());
  }

  private static void sleep(final long delay) {
    try {
      Thread.sleep(delay);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
    }
  }

  private static void printHits(final List<RagHit> hits) {
    if (hits.isEmpty()) {
      System.err.println("(no RAG hits)");
      return;
    }

    System.err.println("RAG hits:");
    int index = 1;
    for (RagHit hit : hits) {
      System.err.printf(ROOT, "  [%d] %.3f  %s%n      %s%n",
        index++, hit.score(), hit.chunk().source(), hit.chunk().text().strip());
    }
    System.err.flush();
  }

  private static final class EmbedProgress {
    private static final int BAR_WIDTH = 24;
    private static final int LINE_WIDTH = 72;

    private final int total;
    private final long startNanos = System.nanoTime();
    private final AtomicBoolean finished = new AtomicBoolean();
    private final ReentrantLock paintLock = new ReentrantLock();

    EmbedProgress(final int total) {
      this.total = Math.max(1, total);
      this.embedded(0);
    }

    private static String formatEta(final double seconds) {
      if (seconds < 60) {
        return String.format(ROOT, "%.0fs", seconds);
      }
      return String.format(ROOT, "%dm%02ds", (int) (seconds / 60), (int) (seconds % 60));
    }

    private static String pad(final String line) {
      return line.length() >= LINE_WIDTH ? line :
        String.format(ROOT, "%-" + LINE_WIDTH + "s", line);
    }

    void embedded(final int done) {
      this.paintLock.lock();
      try {
        if (this.finished.get()) {
          return;
        }
        this.printBar(Math.clamp(done, 0, this.total));
        if (done >= this.total) {
          this.completeBar();
        }
      } finally {
        this.paintLock.unlock();
      }
    }

    void finish() {
      this.paintLock.lock();
      try {
        this.completeBar();
      } finally {
        this.paintLock.unlock();
      }
    }

    private void completeBar() {
      if (!this.finished.compareAndSet(false, true)) {
        return;
      }
      System.out.printf(ROOT, "\r%-" + LINE_WIDTH + "s%n",
        String.format(ROOT, "Embedded: done in %.1fs",
          (System.nanoTime() - this.startNanos) / 1e9));
      System.out.flush();
    }

    private void printBar(final int done) {
      double fraction = (double) done / this.total;
      int filled = (int) Math.round(fraction * BAR_WIDTH);
      String bar = "=".repeat(Math.max(0, filled)) + " ".repeat(Math.max(0, BAR_WIDTH - filled));
      String eta = done <= 0
        ? "--"
        : formatEta((System.nanoTime() - this.startNanos) / 1e9 * (this.total - done) / done);

      System.out.print("\r" + pad(String.format(ROOT, "Embedded: [%s] %3.0f%% (%d/%d) ETA %s",
        bar, fraction * 100.0, done, this.total, eta)));
      System.out.flush();
    }
  }
}
