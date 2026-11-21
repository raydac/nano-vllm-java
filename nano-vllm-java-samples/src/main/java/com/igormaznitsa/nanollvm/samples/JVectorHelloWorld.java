package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
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
 * JVector search over {@code rag/}, then a short chat answer from those passages.
 *
 * <p>The embedding model fills an in-memory graph ({@link JVectorRagIndex}). The chat model
 * writes the reply. The encoder stays open until the questions finish.
 *
 * <p>Args: optional chat directory (default {@code models/Gemma3-270M}), optional embedding
 * file or folder (default gte-small, otherwise the first of multilingual-e5-small,
 * gte-small, or xlm-roberta-base). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.JVectorHelloWorld}
 *
 * @since 1.5.0
 */
public final class JVectorHelloWorld {

  private JVectorHelloWorld() {
  }

  public static void main(final String[] args) {
    Path chatDir = chatModel(args);
    Path encoderDir = encoderModel(args);
    Path ragDir = BundledRag.require();

    System.out.println("Loading chat model from " + chatDir);
    System.out.println("Loading embedding model from " + encoderDir);
    System.out.println("JVector corpus from " + ragDir);

    long started = System.currentTimeMillis();
    try (LlmModel encoder = LlmModelFactory.make(encoderDir);
         LlmModel chat = LlmModelFactory.make(chatDir);
         LLM llm = LLM.builder(chat)
           .noSystemPrompt()
           .maxModelLen(2048)
           .build()) {
      PreparedRag corpus = RagFactory.make(ragDir, RagLoadOptions.forTinyModels());
      JVectorRagIndex index = JVectorRagIndex.of(corpus, encoder);
      System.out.println("JVector chunks: " + index.size());

      RagSession rag = llm.rag(index, 128)
        .topK(2)
        .maxContextChars(900)
        .isolateGeneration(true)
        .sampling(SampleChatPrompts.samplingForDemo(llm.tokenizer(), 128))
        .recoverUnusableAnswers(true)
        .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
        .streamTo(System.err, System.out, false);

      ask(rag, "What are names of the Grimm brothers?");
      ask(rag, "Who are the Grimm brothers' father?");
      ask(rag, "When died the Grimm brothers?");
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  static Path chatModel(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return Path.of(args[0]).toAbsolutePath().normalize();
    }
    return BundledModels.require(BundledModels.GEMMA3_270M);
  }

  static Path encoderModel(final String[] args) {
    if (args != null && args.length > 1 && args[1] != null && !args[1].isBlank()) {
      return Path.of(args[1]).toAbsolutePath().normalize();
    }
    return BundledModels.find(BundledModels.GTE_SMALL_GGUF)
      .orElseGet(BundledModels::requireEmbeddingEncoder);
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
