package io.nanovllm;

import io.nanovllm.engine.ModelRunner;
import io.nanovllm.engine.Scheduler;
import io.nanovllm.engine.Sequence;
import io.nanovllm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class LLM implements AutoCloseable {

  private final Config config;
  private final ModelRunner modelRunner;
  private final Tokenizer tokenizer;
  private final Scheduler scheduler;

  public LLM(String modelPath) {
    this(Path.of(modelPath), Map.of());
  }

  public LLM(Path modelPath) {
    this(modelPath, Map.of());
  }

  public LLM(String modelPath, boolean enforceEager, int tensorParallelSize) {
    this(Path.of(modelPath), Map.of(
        "enforce_eager", enforceEager,
        "tensor_parallel_size", tensorParallelSize
    ));
  }

  public LLM(Path model, Map<String, Object> kwargs) {
    Config.Builder builder = Config.builder(model);
    if (kwargs != null) {
      applyKwargs(builder, kwargs);
    }
    this.config = builder.build();
    Sequence.setBlockSize(this.config.kvcacheBlockSize());
    this.modelRunner = new ModelRunner(this.config);
    this.tokenizer = Tokenizer.fromPretrained(this.config.model());
    if (this.config.eos() < 0) {
      this.config.setEos(this.tokenizer.eosTokenId());
    }
    this.config.setStopTokenIds(this.tokenizer.stopTokenIds());
    this.scheduler = new Scheduler(this.config);
  }

  private static void applyKwargs(Config.Builder builder, Map<String, Object> kwargs) {
    for (var e : kwargs.entrySet()) {
      switch (e.getKey()) {
        case "max_num_batched_tokens", "maxNumBatchedTokens" ->
            builder.maxNumBatchedTokens(((Number) e.getValue()).intValue());
        case "max_num_seqs", "maxNumSeqs" -> builder.maxNumSeqs(((Number) e.getValue()).intValue());
        case "max_model_len", "maxModelLen" ->
            builder.maxModelLen(((Number) e.getValue()).intValue());
        case "gpu_memory_utilization", "gpuMemoryUtilization" ->
            builder.gpuMemoryUtilization(((Number) e.getValue()).floatValue());
        case "tensor_parallel_size", "tensorParallelSize" ->
            builder.tensorParallelSize(((Number) e.getValue()).intValue());
        case "enforce_eager", "enforceEager" -> builder.enforceEager((Boolean) e.getValue());
        case "kvcache_block_size", "kvcacheBlockSize" ->
            builder.kvcacheBlockSize(((Number) e.getValue()).intValue());
        case "num_kvcache_blocks", "numKvcacheBlocks" ->
            builder.numKvcacheBlocks(((Number) e.getValue()).intValue());
        default -> {
        }
      }
    }
  }

  public void addRequest(String prompt, SamplingParams samplingParams) {
    this.addRequest(this.tokenizer.encode(prompt), samplingParams);
  }

  public void addRequest(List<Integer> promptTokenIds, SamplingParams samplingParams) {
    this.scheduler.add(new Sequence(promptTokenIds, samplingParams));
  }

  public StepResult step() {
    Scheduler.ScheduleResult scheduled = this.scheduler.schedule();
    int numTokens = scheduled.prefill()
        ? scheduled.sequences().stream().mapToInt(Sequence::numScheduledTokens).sum()
        : -scheduled.sequences().size();
    @SuppressWarnings("unchecked")
    List<Integer> tokenIds =
        (List<Integer>) this.modelRunner.call("run", scheduled.sequences(), scheduled.prefill());
    List<int[]> appended = new ArrayList<>();
    this.scheduler.postprocess(scheduled.sequences(), tokenIds, scheduled.prefill(), appended);
    List<FinishedOutput> outputs = new ArrayList<>();
    for (Sequence seq : scheduled.sequences()) {
      if (seq.isFinished()) {
        outputs.add(new FinishedOutput(seq.seqId(), seq.completionTokenIds()));
      }
    }
    List<TokenEvent> events = new ArrayList<>(appended.size());
    for (int[] pair : appended) {
      events.add(new TokenEvent(pair[0], pair[1]));
    }
    return new StepResult(outputs, events, numTokens);
  }

  public boolean isFinished() {
    return this.scheduler.isFinished();
  }

  public List<GenerationOutput> generate(List<?> prompts, SamplingParams samplingParams) {
    return this.generate(prompts, samplingParams, true);
  }

  public List<GenerationOutput> generate(List<?> prompts, Object samplingParams, boolean useTqdm) {
    return this.generate(prompts, samplingParams, useTqdm, null);
  }

  public List<GenerationOutput> generate(
      List<?> prompts,
      Object samplingParams,
      boolean useTqdm,
      java.util.function.IntConsumer onToken
  ) {
    List<SamplingParams> params;
    if (samplingParams instanceof SamplingParams sp) {
      params = java.util.Collections.nCopies(prompts.size(), sp);
    } else if (samplingParams instanceof List<?> list) {
      params = new ArrayList<>();
      for (Object o : list) {
        params.add((SamplingParams) o);
      }
    } else {
      throw new IllegalArgumentException(
          "samplingParams must be SamplingParams or List<SamplingParams>");
    }

    for (int i = 0; i < prompts.size(); i++) {
      Object prompt = prompts.get(i);
      if (prompt instanceof String s) {
        this.addRequest(s, params.get(i));
      } else if (prompt instanceof List<?> ids) {
        List<Integer> tokenIds = new ArrayList<>(ids.size());
        for (Object id : ids) {
          tokenIds.add(((Number) id).intValue());
        }
        this.addRequest(tokenIds, params.get(i));
      } else {
        throw new IllegalArgumentException("prompt must be String or List<Integer>");
      }
    }

    Map<Integer, List<Integer>> outputs = new HashMap<>();
    long t0 = System.nanoTime();
    int completed = 0;
    while (!this.isFinished()) {
      StepResult step = this.step();
      if (onToken != null) {
        for (TokenEvent ev : step.tokenEvents()) {
          onToken.accept(ev.tokenId());
        }
      }
      for (FinishedOutput fo : step.outputs()) {
        outputs.put(fo.seqId(), fo.tokenIds());
        completed++;
        if (useTqdm) {
          double elapsed = (System.nanoTime() - t0) / 1e9;
          System.out.printf("\rGenerating %d/%d (%.1fs)", completed, prompts.size(), elapsed);
        }
      }
    }
    if (useTqdm) {
      System.out.println();
    }

    List<GenerationOutput> result = new ArrayList<>(prompts.size());
    List<Integer> sortedIds = outputs.keySet().stream().sorted().toList();
    for (int seqId : sortedIds) {
      List<Integer> tokenIds = outputs.get(seqId);
      result.add(new GenerationOutput(this.tokenizer.decode(tokenIds, true), tokenIds));
    }
    return result;
  }

  public Tokenizer tokenizer() {
    return this.tokenizer;
  }

  public Config config() {
    return this.config;
  }

  @Override
  public void close() {
    this.modelRunner.call("exit");
  }

  public record FinishedOutput(int seqId, List<Integer> tokenIds) {
  }

  public record TokenEvent(int seqId, int tokenId) {
  }

  public record StepResult(List<FinishedOutput> outputs, List<TokenEvent> tokenEvents,
                           int numTokens) {
    public StepResult(List<FinishedOutput> outputs, int numTokens) {
      this(outputs, List.of(), numTokens);
    }
  }

  public record GenerationOutput(String text, List<Integer> tokenIds) {
  }
}
