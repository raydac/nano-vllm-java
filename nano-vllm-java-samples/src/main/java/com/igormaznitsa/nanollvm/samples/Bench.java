package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Command-line throughput smoke test for the inference engine.
 *
 * <p>Loads a model via {@link BundledModels#resolveDefault(String[])}, builds an {@link LLM},
 * and runs one batched {@link LLM#generateTokenIds} over random token-id prompts (quality is
 * irrelevant; the goal is wall-clock tok/s).
 *
 * <p>Args: optional model path/name; optional sequence count (default {@code 8}).
 * Typical: {@code MAVEN_OPTS="-Xmx8g" mvn -pl nano-vllm-java-samples -q exec:java
 * -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Bench
 * -Dexec.args="models/Qwen3-0.6B 8"}
 */
public final class Bench {

  private static final int MAX_MODEL_LEN = 512;
  private static final int KV_BLOCK_SIZE = 256;
  private static final int MAX_INPUT_LEN = 128;
  private static final int MAX_OUTPUT_LEN = 64;

  private Bench() {
  }

  public static void main(final String[] args) {
    Path path = BundledModels.resolveDefault(args);
    int numSeqs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
    int blocksPerSeq = (MAX_MODEL_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    int kvBlocks = Math.max(1, numSeqs) * blocksPerSeq;

    System.out.println("Loading model from " + path);
    System.out.println(
      "Architecture: auto from config.json (override -Dnanollvm.arch=qwen3|gemma3)");
    System.out.printf(
      "Bench: %d seqs, maxModelLen=%d, kvBlocks=%d (heap max %.1f GiB)%n",
      numSeqs,
      MAX_MODEL_LEN,
      kvBlocks,
      Runtime.getRuntime().maxMemory() / (1024.0 * 1024.0 * 1024.0));

    ThreadLocalRandom rnd = ThreadLocalRandom.current();
    try (LlmModel model = LlmModelFactory.make(path, LlmListeners.toSystem());
         LLM llm = LLM.builder(model)
           .maxModelLen(MAX_MODEL_LEN)
           .maxNumSeqs(numSeqs)
           .kvcacheBlockSize(KV_BLOCK_SIZE)
           .numKvcacheBlocks(kvBlocks)
           .listen(LlmListeners.toSystem())
           .build()) {

      List<List<Integer>> prompts = IntStream.range(0, numSeqs)
        .mapToObj(i -> {
          int len = rnd.nextInt(16, MAX_INPUT_LEN + 1);
          return IntStream.range(0, len)
            .map(t -> rnd.nextInt(0, 10_000))
            .boxed()
            .toList();
        })
        .toList();
      List<SamplingParams> params = IntStream.range(0, numSeqs)
        .mapToObj(i -> SamplingParams.builder()
          .temperature(0.6f)
          .maxTokens(rnd.nextInt(8, MAX_OUTPUT_LEN + 1))
          .ignoreEos(true)
          .build())
        .toList();

      long started = System.nanoTime();
      llm.generateTokenIds(prompts, params, false, Duration.ZERO, null);
      double seconds = (System.nanoTime() - started) / 1e9;
      int totalTokens = params.stream().mapToInt(SamplingParams::maxTokens).sum();
      System.out.printf(
        "Total: %dtok, Time: %.2fs, Throughput: %.2ftok/s%n",
        totalTokens,
        seconds,
        totalTokens / seconds);
    }
  }
}
