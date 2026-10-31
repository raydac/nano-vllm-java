package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BenchProfiler;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Command-line throughput smoke test for the inference engine.
 *
 * <p>Loads a model via {@link BundledModels#resolveDefault(String[])}, builds an {@link LLM},
 * and runs one batched {@link LLM#generateTokenIds} over random token-id prompts (quality is
 * irrelevant; the goal is wall-clock tok/s).
 *
 * <p>Args: optional model path/name; optional sequence count (default {@code 8}).
 * Typical: {@code MAVEN_OPTS="-Xmx8g" mvn22 -pl nano-vllm-java-samples -q exec:java
 * -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Bench
 * -Dexec.args="models/Qwen3-0.6B 8"}
 *
 * <p>Profile that generate with async-profiler (scalar by default):
 * {@code mvn22 -pl nano-vllm-java-samples -am -Pbench package exec:exec@bench}.
 * {@code -Dbench.kernels=vector} selects the Vector API. {@code -Dbench.seqs},
 * {@code -Dbench.maxInput}, and {@code -Dbench.maxOutput} shorten the run. Reports are written
 * under {@code nano-vllm-java-samples/target/bench-profile/}.
 */
public final class Bench {

  static final String PROP_MAX_INPUT = "nanollvm.bench.maxInput";
  static final String PROP_MAX_OUTPUT = "nanollvm.bench.maxOutput";

  private static final int MAX_MODEL_LEN = 512;
  private static final int KV_BLOCK_SIZE = 256;
  private static final int MAX_INPUT_LEN = 128;
  private static final int MAX_OUTPUT_LEN = 64;
  private static final int MIN_INPUT_LEN = 16;
  private static final int MIN_OUTPUT_LEN = 8;

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
    int maxInput = inclusiveLimit(PROP_MAX_INPUT, MAX_INPUT_LEN, MIN_INPUT_LEN);
    int maxOutput = inclusiveLimit(PROP_MAX_OUTPUT, MAX_OUTPUT_LEN, MIN_OUTPUT_LEN);
    System.out.printf(
      Locale.ROOT,
      "Bench: %d seqs, maxModelLen=%d, input<=%d, output<=%d, kvBlocks=%d (heap max %.1f GiB)%n",
      numSeqs,
      MAX_MODEL_LEN,
      maxInput,
      maxOutput,
      kvBlocks,
      Runtime.getRuntime().maxMemory() / (1024.0 * 1024.0 * 1024.0));

    Random rnd = new Random();
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
          int len = MIN_INPUT_LEN + rnd.nextInt(maxInput - MIN_INPUT_LEN + 1);
          return IntStream.range(0, len)
            .map(t -> rnd.nextInt(10_000))
            .boxed()
            .toList();
        })
        .toList();
      List<SamplingParams> params = IntStream.range(0, numSeqs)
        .mapToObj(i -> SamplingParams.builder()
          .temperature(0.6f)
          .maxTokens(MIN_OUTPUT_LEN + rnd.nextInt(maxOutput - MIN_OUTPUT_LEN + 1))
          .ignoreEos(true)
          .build())
        .toList();

      try (BenchProfiler profile = BenchProfiler.open()) {
        long started = System.nanoTime();
        llm.generateTokenIds(prompts, params, false, Duration.ZERO, null);
        double seconds = (System.nanoTime() - started) / 1e9;
        int totalTokens = params.stream().mapToInt(SamplingParams::maxTokens).sum();
        System.out.printf(
          Locale.ROOT,
          "Total: %dtok, Time: %.2fs, Throughput: %.2ftok/s%n",
          totalTokens,
          seconds,
          totalTokens / seconds);
      }
    }
  }

  static int inclusiveLimit(final String key, final int fallback, final int minimum) {
    String raw = NanoLlvmProps.systemProperty(key);
    int value = raw == null ? fallback : Integer.parseInt(raw.strip());
    if (value < minimum) {
      throw new IllegalArgumentException("%s must be >= %d".formatted(key, minimum));
    }
    return value;
  }
}
