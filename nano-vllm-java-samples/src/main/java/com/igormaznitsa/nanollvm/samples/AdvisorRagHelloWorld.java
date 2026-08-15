package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Minimal BM25 RAG + custom advisor demo (default Gemma3-270M).
 *
 * <p>Wires advisor {@code Alex} on the engine, indexes {@code rag/} with lexical BM25, then asks
 * two Grimm questions with no extra prompt wrapping.
 *
 * <p>Args: optional model directory (default {@code models/Gemma3-270M} via
 * {@link BundledModels}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.AdvisorRagHelloWorld}
 *
 * @since 1.1.0
 */
public final class AdvisorRagHelloWorld {

  private AdvisorRagHelloWorld() {
  }

  public static void main(final String[] args) {
    Path modelDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.GEMMA3_270M);
    Path ragDir = BundledRag.require();

    System.out.println("Loading model from " + modelDir);
    System.out.println("BM25 corpus from " + ragDir);

    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model)
           .noSystemPrompt()
           .maxModelLen(2048)
           .advisors(LlmAdvisorMixer.defaults(),
             LlmAdvisor.builder()
               .name("Alex")
               .prompt("Find facts and shortly present them.")
               .build())
           .advisorNoteFilter(note -> !SampleChatPrompts.isSetupBoilerplate(note))
           .build()) {
      PreparedRag corpus = RagFactory.make(ragDir, RagLoadOptions.forTinyModels());
      System.out.println("BM25 chunks: " + corpus.size());

      RagSession rag = llm.rag(corpus, 128)
        .topK(2)
        .maxContextChars(900)
        .isolateGeneration(true)
        .sampling(SampleChatPrompts.samplingForDemo(llm.tokenizer(), 128));
      rag.chat()
        .recoverUnusableAnswers(true)
        .unusableAnswer(SampleChatPrompts::isSetupBoilerplate);
      rag.streamTo(System.err, System.out, false);

      ask(rag, "What are names of the Grimm brothers?");
      ask(rag, "Who are the Grimm brothers' father?");
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static void ask(final RagSession rag, final String question) {
    System.out.println();
    System.out.println("Q: " + question);
    rag.send(question);
    System.out.println();
    for (RagHit hit : rag.lastHits()) {
      System.out.printf(Locale.ROOT, "  %.3f  %s%n", hit.score(), hit.chunk().source());
    }
  }
}
