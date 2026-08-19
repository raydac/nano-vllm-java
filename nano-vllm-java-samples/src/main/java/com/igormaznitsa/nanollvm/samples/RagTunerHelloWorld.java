package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.rag.RagTuner;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.EpubText;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * BM25 RAG over a bundled EPUB: {@link RagTuner} filter / extract / preprocess plus
 * <a href="https://github.com/documentnode/epub4j">epub4j</a> (fork of
 * <a href="https://github.com/psiegman/epublib">epublib</a>) for the book text.
 *
 * <p>Indexes Project Gutenberg {@code pg59112.epub} (Karel Čapek, <em>R.U.R.</em>) from the
 * samples classpath, then asks two questions that are answered in that play.
 *
 * <p>Args: optional model directory (default {@code models/Gemma3-270M} via
 * {@link BundledModels}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.RagTunerHelloWorld}
 *
 * @since 1.1.1
 */
public final class RagTunerHelloWorld {

  private RagTunerHelloWorld() {
  }

  public static void main(final String[] args) {
    Path modelDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.QWEN3_0_6B);

    System.out.println("Loading model from " + modelDir);
    System.out.println("EPUB corpus classpath:" + EpubText.RUR_EPUB);

    long started = System.currentTimeMillis();
    PreparedRag corpus = indexBundledEpub();
    System.out.println("BM25 chunks: " + corpus.size());

    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model)
           .noSystemPrompt()
           .maxModelLen(2048)
           .build()) {
      RagSession rag = llm.rag(corpus, 128)
        .topK(3)
        .maxContextChars(900)
        .isolateGeneration(true)
        .sampling(SampleChatPrompts.samplingForDemo(llm.tokenizer(), 128))
        .recoverUnusableAnswers(true)
        .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
        .streamTo(System.err, System.out, false);

      ask(rag, "What does the Czech word robot mean?");
      ask(rag, "Who is Helena Glory?");
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  static PreparedRag indexBundledEpub() {
    return RagFactory.builder()
      .forTinyModels()
      .addProcessor(
        RagTuner.allowing(EpubText::isEpub),
        RagTuner.extracting(EpubText::extract),
        RagTuner.preprocessing(EpubText::normalizeWhitespace))
      .addResource(RagTunerHelloWorld.class, "/" + EpubText.RUR_EPUB)
      .build();
  }

  private static void ask(final RagSession rag, final String question) {
    System.out.println();
    System.out.println("Q: " + question);
    rag.send(question);
    System.out.println();
    printHits(rag.lastHits());
  }

  private static void printHits(final List<RagHit> hits) {
    if (hits.isEmpty()) {
      System.out.println("(no RAG hits)");
      return;
    }

    System.out.println("RAG hits:");
    int index = 1;
    for (RagHit hit : hits) {
      System.out.printf(Locale.ROOT, "  [%d] %.3f  %s%n",
        index++, hit.score(), hit.chunk().source());
      System.out.println("      " + hit.chunk().text().strip());
    }
  }
}
