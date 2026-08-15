package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Minimal next-token continuation demo (default Tiny-LLM-ONNX): encode a seed, sample the next
 * several tokens, print each id plus piece, then the continued text.
 *
 * <p>Tiny-LLM is a ~10M base checkpoint — fast to load, meant for this API, not for chat quality.
 * Pass any other causal folder as the first argument if you want a stronger continuation.
 *
 * <p>Args: optional model directory (default {@code models/Tiny-LLM-ONNX}), optional seed text
 * (default {@code The capital of France is}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.NextTokenHelloWorld}
 *
 * @since 1.1.0
 */
public final class NextTokenHelloWorld {

  private static final int NEXT_TOKENS = 8;
  private static final String DEFAULT_SEED = "The capital of France is";

  private NextTokenHelloWorld() {
  }

  public static void main(final String[] args) {
    Path modelDir = modelPath(args);
    String seed = seedText(args);

    System.out.println("Loading model from " + modelDir);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model)
           .disableMultiCpu()
           .maxModelLen(256)
           .build()) {
      continueSeed(llm, seed);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static void continueSeed(final LLM llm, final String seed) {
    Tokenizer tokenizer = llm.tokenizer();
    List<Integer> promptIds = tokenizer.encode(seed);
    SamplingParams sampling = SamplingParams.builder()
      .temperature(0.2f)
      .maxTokens(NEXT_TOKENS)
      .build();

    System.out.println("seed: " + seed);
    System.out.println("prompt tokens: " + promptIds);
    System.out.println("next " + NEXT_TOKENS + " tokens:");

    LLM.GenerationOutput output = llm.generateTokenIds(
        List.of(promptIds),
        sampling,
        (LLM.TokenEvent event) -> printToken(tokenizer, event.tokenId()))
      .getFirst();

    System.out.println("continuation: " + output.text());
    System.out.println("full text: " + seed + output.text());
  }

  private static void printToken(final Tokenizer tokenizer, final int tokenId) {
    System.out.printf(Locale.ROOT, "  %d\t%s%n", tokenId, tokenizer.decode(List.of(tokenId)));
  }

  private static Path modelPath(final String[] args) {
    return cliArg(args, 0)
      .map(path -> Path.of(path).toAbsolutePath().normalize())
      .orElseGet(() -> BundledModels.require(BundledModels.TINY_LLM_ONNX));
  }

  private static String seedText(final String[] args) {
    if (args == null || args.length < 2) {
      return DEFAULT_SEED;
    }
    String joined = Arrays.stream(args, 1, args.length)
      .filter(part -> part != null && !part.isBlank())
      .collect(joining(" "));
    return joined.isBlank() ? DEFAULT_SEED : joined;
  }

  private static Optional<String> cliArg(final String[] args, final int index) {
    if (args == null || args.length <= index || args[index] == null || args[index].isBlank()) {
      return Optional.empty();
    }
    return Optional.of(args[index]);
  }
}
