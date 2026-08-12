package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;

import java.nio.file.Path;

/**
 * Minimal Gemma3 demo: say hello and print the reply.
 *
 * <p>Args: optional model directory (default {@code models/Gemma3-270M} via
 * {@link BundledModels}). Example:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.HelloWorld}
 */
public final class HelloWorld {

  private HelloWorld() {
  }

  public static void main(final String[] args) {
    Path gemmaDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.GEMMA3_270M);

    System.out.println("Loading Gemma3 from " + gemmaDir);

    final long time = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(gemmaDir);
         LLM llm = LLM.builder(model)
           .noSystemPrompt()
           .build()) {
      String reply = llm.chatOnce("Hello!");
      System.out.println(reply);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - time) + "ms");
    }
  }
}
