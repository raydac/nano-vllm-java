package com.igormaznitsa.nanollvm.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads {@code *.safetensors} into an immutable {@link WeightBag} (packed Q/K/V and gate/up
 * shards are merged before the bag is returned).
 */
public final class ModelLoader {

  private ModelLoader() {
  }

  public static WeightBag loadWeights(
    final Path modelDir,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    List<Path> files = SafetensorsReader.listSafetensors(modelDir);
    if (files.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors files in " + modelDir);
    }

    long fileBytes = 0L;
    for (Path file : files) {
      fileBytes += Files.size(file);
    }
    LlmListeners.infof(streams, null, "Loading weights from %s (%.2f GiB, %d file%s)%n",
      modelDir,
      fileBytes / (1024.0 * 1024.0 * 1024.0),
      files.size(),
      files.size() == 1 ? "" : "s");

    List<WeightSource> sources = files.stream()
      .map(file -> new WeightSource(PathNames.of(file), () -> SafetensorsReader.open(file)))
      .toList();
    return assembleWeights(sources, modelDir.toString(), hfConfig, schema, streams);
  }

  /**
   * Assembles a {@link WeightBag} from in-memory safetensors blobs (classpath / {@link ModelFileBundle}).
   *
   * @since 1.1.0
   */
  public static WeightBag loadWeights(
    final List<ModelFileBundle.NamedBytes> blobs,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    if (blobs.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors blobs for " + label);
    }
    long fileBytes = blobs.stream().mapToLong(b -> b.bytes().length).sum();
    LlmListeners.infof(streams, null, "Loading weights from %s (%.2f GiB, %d file%s)%n",
      label,
      fileBytes / (1024.0 * 1024.0 * 1024.0),
      blobs.size(),
      blobs.size() == 1 ? "" : "s");
    List<WeightSource> sources = blobs.stream()
      .map(blob -> new WeightSource(
        blob.name(),
        () -> SafetensorsReader.open(blob.buffer(), blob.name())))
      .toList();
    return assembleWeights(sources, label, hfConfig, schema, streams);
  }

  private static WeightBag assembleWeights(
    final List<WeightSource> sources,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener streams
  ) throws IOException {
    Map<String, Object[]> packed = schema.packedModulesMapping();
    List<PlannedTensor> plan = new ArrayList<>();
    for (WeightSource source : sources) {
      try (SafetensorsReader probe = source.open()) {
        for (String weightName : probe.keys()) {
          ResolvedParam resolved = resolveParam(weightName, schema, packed);
          if (resolved == null) {
            continue;
          }
          plan.add(new PlannedTensor(source, weightName, resolved.paramName(), resolved.shardId(),
            probe.byteSize(weightName)));
        }
      }
    }
    if (plan.isEmpty()) {
      throw new IllegalStateException("no matching weight tensors found in " + label);
    }

    WeightAssembler assembler = new WeightAssembler(hfConfig);
    LoadProgress progress = new LoadProgress("Weights", plan.size(), streams);
    long loadedBytes = 0L;
    WeightSource current = null;
    SafetensorsReader reader = null;
    try {
      for (PlannedTensor item : plan) {
        if (reader == null || !item.source().equals(current)) {
          if (reader != null) {
            reader.close();
          }
          current = item.source();
          reader = current.open();
        }
        Tensor tensor = reader.getTensor(item.weightName());
        assembler.accept(item.paramName(), item.shardId(), tensor);
        loadedBytes += item.byteSize();
        progress.step("%s | %s (%.0f MiB)".formatted(
          item.source().label(),
          item.weightName(),
          loadedBytes / (1024.0 * 1024.0)));
      }
      progress.finish("%.0f MiB loaded".formatted(loadedBytes / (1024.0 * 1024.0)));
    } catch (RuntimeException | IOException e) {
      progress.finish("failed");
      throw e;
    } finally {
      if (reader != null) {
        reader.close();
      }
    }

    WeightBag bag = assembler.build();
    for (String required : schema.expectedParameters()) {
      if (!bag.has(required)) {
        throw new IllegalStateException("missing required weight after load: " + required);
      }
    }
    return bag;
  }

  /**
   * Assembles a {@link WeightBag} from already-decoded float tensors (ONNX / other formats).
   *
   * @since 1.1.0
   */
  public static WeightBag assembleFromNamedTensors(
    final Map<String, Tensor> namedTensors,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) {
    requireNonNull(namedTensors, "namedTensors");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    if (namedTensors.isEmpty()) {
      throw new IllegalArgumentException("no weight tensors for " + label);
    }
    Map<String, Object[]> packed = schema.packedModulesMapping();
    WeightAssembler assembler = new WeightAssembler(hfConfig);
    int matched = 0;
    for (var e : namedTensors.entrySet()) {
      ResolvedParam resolved = resolveParam(e.getKey(), schema, packed);
      if (resolved == null) {
        continue;
      }
      assembler.accept(resolved.paramName(), resolved.shardId(), e.getValue());
      matched++;
    }
    if (matched == 0) {
      throw new IllegalStateException("no matching weight tensors found in " + label);
    }
    LlmListeners.infof(streams, null, "Assembled %d weight tensors from %s%n", matched, label);
    WeightBag bag = assembler.build();
    for (String required : schema.expectedParameters()) {
      if (!bag.has(required)) {
        throw new IllegalStateException("missing required weight after load: " + required);
      }
    }
    return bag;
  }

  private static ResolvedParam resolveParam(
    final String weightName,
    final WeightSchema schema,
    final Map<String, Object[]> packed
  ) {
    for (var e : packed.entrySet()) {
      String key = e.getKey();
      if (weightName.contains(key)) {
        Object[] mapping = e.getValue();
        String paramName = weightName.replace(key, (String) mapping[0]);
        if (!schema.accepts(paramName)) {
          return null;
        }
        return new ResolvedParam(paramName, mapping[1]);
      }
    }
    if (!schema.accepts(weightName)) {
      return null;
    }
    return new ResolvedParam(weightName, null);
  }

  private interface ShardMerge {
    void write(final Object shardId, final Tensor loaded);

    Tensor finish();
  }

  @FunctionalInterface
  private interface ReaderOpen {
    SafetensorsReader open() throws IOException;
  }

  private record ResolvedParam(String paramName, Object shardId) {
  }

  private record PlannedTensor(
    WeightSource source,
    String weightName,
    String paramName,
    Object shardId,
    long byteSize
  ) {
  }

  private record WeightSource(String label, ReaderOpen opener) {
    SafetensorsReader open() throws IOException {
      return this.opener.open();
    }
  }

  private static final class WeightAssembler {
    private final Config.HfConfig hf;
    private final Map<String, Tensor> complete = new LinkedHashMap<>();
    private final Map<String, ShardMerge> merges = new HashMap<>();

    WeightAssembler(Config.HfConfig hf) {
      this.hf = hf;
    }

    void accept(final String paramName, final Object shardId, final Tensor tensor) {
      if (shardId == null) {
        this.complete.put(paramName, tensor);
        return;
      }
      this.merges.computeIfAbsent(paramName, this::newMerge).write(shardId, tensor);
    }

    WeightBag build() {
      for (var e : this.merges.entrySet()) {
        this.complete.put(e.getKey(), e.getValue().finish());
      }
      return new WeightBag(this.complete);
    }

    private ShardMerge newMerge(final String paramName) {
      if (paramName.contains(QKV_PROJ)) {
        return new QkvMerge(this.hf);
      }
      if (paramName.contains(GATE_UP_PROJ)) {
        return new GateUpMerge(this.hf);
      }
      throw new IllegalArgumentException("unknown packed weight target: " + paramName);
    }
  }

  private static final class QkvMerge implements ShardMerge {
    private final Tensor weight;
    private final int headSize;
    private final int numHeads;
    private final int numKvHeads;
    private final boolean[] seen = new boolean[3];

    QkvMerge(Config.HfConfig hf) {
      this.headSize = hf.headDim();
      this.numHeads = hf.numAttentionHeads();
      this.numKvHeads = hf.numKeyValueHeads();
      int out = (this.numHeads + 2 * this.numKvHeads) * this.headSize;
      this.weight = Tensor.zeros(out, hf.hiddenSize());
    }

    @Override
    public void write(final Object shardId, final Tensor loaded) {
      String id = String.valueOf(shardId);
      int shardSize;
      int shardOffset;
      int seenIndex;
      switch (id) {
        case "q" -> {
          shardSize = this.numHeads * this.headSize;
          shardOffset = 0;
          seenIndex = 0;
        }
        case "k" -> {
          shardSize = this.numKvHeads * this.headSize;
          shardOffset = this.numHeads * this.headSize;
          seenIndex = 1;
        }
        case "v" -> {
          shardSize = this.numKvHeads * this.headSize;
          shardOffset = this.numHeads * this.headSize + this.numKvHeads * this.headSize;
          seenIndex = 2;
        }
        default -> throw new IllegalArgumentException("shardId " + shardId);
      }
      this.copyRows(loaded, shardOffset, shardSize);
      this.seen[seenIndex] = true;
    }

    private void copyRows(final Tensor loaded, final int shardOffset, final int shardSize) {
      int in = this.weight.size(1);
      for (int o = 0; o < shardSize; o++) {
        System.arraycopy(
          loaded.data(), loaded.offset() + o * in,
          this.weight.data(), (shardOffset + o) * in,
          in);
      }
    }

    @Override
    public Tensor finish() {
      if (!this.seen[0] || !this.seen[1] || !this.seen[2]) {
        throw new IllegalStateException("incomplete qkv_proj merge (need q,k,v shards)");
      }
      return this.weight;
    }
  }

  private static final class GateUpMerge implements ShardMerge {
    private final Tensor weight;
    private final int[] outputSizes;
    private final boolean[] seen;

    GateUpMerge(Config.HfConfig hf) {
      this.outputSizes = new int[] {hf.intermediateSize(), hf.intermediateSize()};
      this.weight = Tensor.zeros(this.outputSizes[0] + this.outputSizes[1], hf.hiddenSize());
      this.seen = new boolean[this.outputSizes.length];
    }

    @Override
    public void write(final Object shardId, final Tensor loaded) {
      int id = ((Number) shardId).intValue();
      if (id < 0 || id >= this.outputSizes.length) {
        throw new IllegalArgumentException("shardId " + shardId);
      }
      int shardOffset = 0;
      for (int i = 0; i < id; i++) {
        shardOffset += this.outputSizes[i];
      }
      int shardSize = this.outputSizes[id];
      int in = this.weight.size(1);
      for (int o = 0; o < shardSize; o++) {
        System.arraycopy(
          loaded.data(), loaded.offset() + o * in,
          this.weight.data(), (shardOffset + o) * in,
          in);
      }
      this.seen[id] = true;
    }

    @Override
    public Tensor finish() {
      for (boolean ok : this.seen) {
        if (!ok) {
          throw new IllegalStateException("incomplete gate_up_proj merge");
        }
      }
      return this.weight;
    }
  }
}
