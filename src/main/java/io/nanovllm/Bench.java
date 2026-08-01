package io.nanovllm;

import io.nanovllm.utils.BundledModels;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

public final class Bench {

  private Bench() {
  }

  public static void main(String[] args) {
    Path path = BundledModels.resolveDefault(args);
    int numSeqs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
    int maxInputLen = 128;
    int maxOutputLen = 64;

    System.out.println("Loading model from " + path);
    System.out.println(
        "Architecture: auto from config.json (override -Dnanovllm.arch=qwen3|gemma3)");
    ThreadLocalRandom rnd = ThreadLocalRandom.current();
    try (LLM llm = new LLM(path, Map.of(
        "enforce_eager", true,
        "max_model_len", 512,
        "max_num_seqs", numSeqs,
        "num_kvcache_blocks", 256
    ))) {
      List<List<Integer>> prompts = new ArrayList<>();
      List<SamplingParams> params = new ArrayList<>();
      for (int i = 0; i < numSeqs; i++) {
        int len = rnd.nextInt(16, maxInputLen + 1);
        List<Integer> ids = new ArrayList<>(len);
        for (int t = 0; t < len; t++) {
          ids.add(rnd.nextInt(0, 10_000));
        }
        prompts.add(ids);
        params.add(new SamplingParams(0.6f, rnd.nextInt(8, maxOutputLen + 1), true));
      }

      long t0 = System.nanoTime();
      llm.generate(prompts, params, false);
      double seconds = (System.nanoTime() - t0) / 1e9;
      int totalTokens = params.stream().mapToInt(SamplingParams::maxTokens).sum();
      System.out.printf("Total: %dtok, Time: %.2fs, Throughput: %.2ftok/s%n",
          totalTokens, seconds, totalTokens / seconds);
    }
  }
}
