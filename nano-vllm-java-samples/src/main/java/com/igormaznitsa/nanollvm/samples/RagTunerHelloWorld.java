package com.igormaznitsa.nanollvm.samples;

import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
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

/**
 * BM25 RAG over a bundled EPUB: {@link RagTuner} filter / extract / preprocess, JDK zip + StAX
 * for the book text, then lexical retrieval.
 *
 * <p>Indexes Project Gutenberg {@code pg59112.epub} (Karel Čapek, <em>R.U.R.</em>) from the
 * samples classpath, then asks questions that are answered in that play. Chat uses
 * {@link LLM.Builder#deterministic()} so repeated runs pick the same tokens. Each generate is
 * isolated from prior answers; EPUB metadata is kept as a heading so later chunks still name
 * the play.
 *
 * <p>Args: optional chat model directory (default {@code models/Qwen3-0.6B}). From the
 * repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.RagTunerHelloWorld}
 *
 * @since 1.2.0
 */
public final class RagTunerHelloWorld {

  private RagTunerHelloWorld() {
  }

  public static void main(final String[] args) {
    Path chatDir = chatModel(args);

    System.out.println("EPUB corpus classpath:" + EpubText.RUR_EPUB);

    long started = System.currentTimeMillis();
    PreparedRag documents = loadBundledEpub();
    System.out.println("BM25 chunks: " + documents.size());

    System.out.println("Loading chat model from " + chatDir);
    try (LlmModel chat = LlmModelFactory.make(chatDir);
         LLM llm = LLM.builder(chat)
           .noSystemPrompt()
           .maxModelLen(2048)
           .deterministic()
           .build()) {
      RagSession rag = llm.rag(documents, 128)
        .topK(3)
        .maxContextChars(1800)
        .isolateGeneration(true)
        .recoverUnusableAnswers(true)
        .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
        .streamTo(System.err, System.out, false);

      ask(rag, "What does the Czech word robot mean?");
      ask(rag, "Who is Helena Glory?");
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
}
