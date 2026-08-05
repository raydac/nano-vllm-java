package com.igormaznitsa.nanollvm;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.utils.BundledModels;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Command-line throughput smoke test for the inference engine.
 *
 * <p>Loads a model via {@link BundledModels#resolveDefault(String[])} (same resolution as
 * {@link Example}), builds an {@link LLM} with CLI progress ({@link LLM.Builder#withSystemIo()}),
 * and runs one batched {@link LLM#generate} call over several concurrent sequences. Prompts are
 * random token-id lists (not real text), so output quality is meaningless; the goal is to stress
 * scheduling, KV cache paging, and the forward path and print wall-clock throughput (tokens per second).
 *
 * <p>Arguments: optional model directory or name (see {@link BundledModels}); optional second
 * argument — number of concurrent sequences (default {@code 8}).
 *
 * <p>Typical launch (weights + KV need a large heap; use at least {@code -Xmx8g} for Qwen3-0.6B):
 * {@code MAVEN_OPTS="-Xmx8g" mvn -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.Bench
 * -Dexec.args="models/Qwen3-0.6B 8"}
 */
public final class Bench {

  private static final int MAX_MODEL_LEN = 512;
  private static final int KV_BLOCK_SIZE = 256;
  private static final int MAX_INPUT_LEN = 128;
  private static final int MAX_OUTPUT_LEN = 64;

  private Bench() {
  }

  public static void main(String[] args) {
    Path path = BundledModels.resolveDefault(args);
    int numSeqs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
    int kvBlocks = kvBlocksFor(numSeqs);

    System.out.println("Loading model from " + path);
    System.out.println(
        "Architecture: auto from config.json (override -Dnanovllm.arch=qwen3|gemma3)");
    System.out.printf("Bench: %d seqs, maxModelLen=%d, kvBlocks=%d (heap max %s)%n",
        numSeqs, MAX_MODEL_LEN, kvBlocks, formatBytes(Runtime.getRuntime().maxMemory()));
    ThreadLocalRandom rnd = ThreadLocalRandom.current();
    try (LLM llm = LLM.builder(path)
        .enforceEager(true)
        .maxModelLen(MAX_MODEL_LEN)
        .maxNumSeqs(numSeqs)
        .kvcacheBlockSize(KV_BLOCK_SIZE)
        .numKvcacheBlocks(kvBlocks)
        .skipWarmup()
        .withSystemIo()
        .build()) {
      List<List<Integer>> prompts = new ArrayList<>();
      List<SamplingParams> params = new ArrayList<>();
      for (int i = 0; i < numSeqs; i++) {
        int len = rnd.nextInt(16, MAX_INPUT_LEN + 1);
        List<Integer> ids = new ArrayList<>(len);
        for (int t = 0; t < len; t++) {
          ids.add(rnd.nextInt(0, 10_000));
        }
        prompts.add(ids);
        params.add(new SamplingParams(0.6f, rnd.nextInt(8, MAX_OUTPUT_LEN + 1), true));
      }

      long t0 = System.nanoTime();
      llm.generate(prompts, params, false);
      double seconds = (System.nanoTime() - t0) / 1e9;
      int totalTokens = params.stream().mapToInt(SamplingParams::maxTokens).sum();
      System.out.printf("Total: %dtok, Time: %.2fs, Throughput: %.2ftok/s%n",
          totalTokens, seconds, totalTokens / seconds);
    }
  }

  private static int kvBlocksFor(int numSeqs) {
    int blocksPerSeq = (MAX_MODEL_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    return Math.max(1, numSeqs) * blocksPerSeq;
  }

  private static String formatBytes(long bytes) {
    return "%.1f GiB".formatted(bytes / (1024.0 * 1024.0 * 1024.0));
  }
}
