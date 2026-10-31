package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.models.llmcontainer.LoadProgress;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsTransport;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Model-layer safetensors decode: match container tensor names to {@link WeightSchema}, merge
 * packed Q/K/V and gate/up shards, then return an immutable {@link WeightBag}.
 *
 * <p>Wrapper prefixes ({@code language_model}, {@code text_model}, {@code base_model},
 * {@code model.model}) are peeled until a schema name matches. {@code attention} /
 * {@code feed_forward} / {@code ffn} map onto {@code self_attn} / {@code mlp} when that is the
 * schema name. A module name matches a packed
 * rule only as a whole path segment, so an already fused {@code gate_up_proj} is left intact.
 * A second copy of the same parameter, or a matrix whose shape is not the config's
 * {@code [out, in]} (or that layout swapped), fails the load.
 *
 * @since 1.1.0
 */
public final class ModelLoader {

  private static final Map<String, String> SEGMENT_ALIASES = Map.of(
    "attention", "self_attn",
    "feed_forward", "mlp",
    "ffn", "mlp"
  );

  private static final String[] WRAPPERS = {
    "base_model.model.",
    "model.language_model.",
    "model.text_model.",
    "language_model.model.",
    "text_model.model.",
    "language_model.",
    "text_model.",
    "model.model."
  };

  private ModelLoader() {
  }

  /**
   * Reads safetensors shards into a bag using the bound schema.
   *
   * @since 1.1.0
   */
  static WeightBag fill(
    final SafetensorsTransport transport,
    final ModelBinding.BoundModel bound,
    final LlmListener streams
  ) throws IOException {
    requireNonNull(transport, "transport");
    requireNonNull(bound, "bound");
    long fileBytes =
      transport.tensorIndex().stream().mapToLong(SafetensorsTransport.TensorRef::byteSize).sum();
    LlmListeners.infof(streams, null, "Loading weights from %s (%.2f GiB)%n",
      transport.label(),
      fileBytes / (1024.0 * 1024.0 * 1024.0));
    return assembleWeights(transport, bound.config(), bound.schema(), streams);
  }

  private static WeightBag assembleWeights(
    final SafetensorsTransport transport,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener streams
  ) throws IOException {
    Map<String, Object[]> packed = schema.packedModulesMapping();
    List<PlannedTensor> plan = new ArrayList<>();
    for (SafetensorsTransport.TensorRef ref : transport.tensorIndex()) {
      ResolvedParam resolved = resolveParam(ref.name(), schema, packed);
      if (resolved == null) {
        continue;
      }
      plan.add(new PlannedTensor(ref, resolved.paramName(), resolved.shardId()));
    }
    if (plan.isEmpty()) {
      throw new IllegalStateException("no matching weight tensors found in " + transport.label());
    }

    WeightAssembler assembler = new WeightAssembler(hfConfig);
    LoadProgress progress = new LoadProgress("Weights", plan.size(), streams);
    long loadedBytes = 0L;
    String currentShard = null;
    SafetensorsReader reader = null;
    try {
      for (PlannedTensor item : plan) {
        String shardLabel = item.ref().shardLabel();
        if (reader == null || !shardLabel.equals(currentShard)) {
          if (reader != null) {
            reader.close();
          }
          currentShard = shardLabel;
          reader = transport.openShard(currentShard);
        }
        Tensor tensor = reader.getTensor(item.ref().name());
        assembler.accept(item.paramName(), item.shardId(), tensor);
        loadedBytes += item.ref().byteSize();
        progress.step("%s | %s (%.0f MiB)".formatted(
          shardLabel,
          item.ref().name(),
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

    return assembler.build(schema);
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
    requireNonNull(label, "label");
    requireNonNull(hfConfig, "hfConfig");
    requireNonNull(schema, "schema");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    if (namedTensors.isEmpty()) {
      throw new IllegalArgumentException("no weight tensors for " + label);
    }
    Map<String, Object[]> packed = schema.packedModulesMapping();
    WeightAssembler assembler = new WeightAssembler(hfConfig);
    int matched = 0;
    for (var entry : namedTensors.entrySet()) {
      ResolvedParam resolved = resolveParam(entry.getKey(), schema, packed);
      if (resolved == null) {
        continue;
      }
      assembler.accept(resolved.paramName(), resolved.shardId(), entry.getValue());
      matched++;
    }
    if (matched == 0) {
      throw new IllegalStateException("no matching weight tensors found in " + label);
    }
    LlmListeners.infof(streams, null, "Assembled %d weight tensors from %s%n", matched, label);
    return assembler.build(schema);
  }

  private static ResolvedParam resolveParam(
    final String weightName,
    final WeightSchema schema,
    final Map<String, Object[]> packed
  ) {
    ResolvedParam found = null;
    for (String candidate : nameCandidates(weightName)) {
      ResolvedParam chosen = preferSchemaName(candidate, schema, packed);
      if (chosen == null) {
        continue;
      }
      if (found != null && !found.equals(chosen)) {
        throw new IllegalStateException("ambiguous weight name '" + weightName + "'");
      }
      found = chosen;
    }
    return found;
  }

  private static ResolvedParam preferSchemaName(
    final String candidate,
    final WeightSchema schema,
    final Map<String, Object[]> packed
  ) {
    ResolvedParam direct = accepted(candidate, schema);
    if (direct != null) {
      return direct;
    }
    String aliased = aliasSegments(candidate);
    if (aliased != null) {
      ResolvedParam aliasedDirect = accepted(aliased, schema);
      if (aliasedDirect != null) {
        return aliasedDirect;
      }
    }
    ResolvedParam packedMatch = matchPacked(aliased != null ? aliased : candidate, schema, packed);
    if (packedMatch != null) {
      return packedMatch;
    }
    return aliased == null ? null : matchPacked(candidate, schema, packed);
  }

  private static ResolvedParam accepted(final String name, final WeightSchema schema) {
    return schema.accepts(name) ? new ResolvedParam(name, null) : null;
  }

  private static String aliasSegments(final String name) {
    String[] parts = name.split("\\.", -1);
    boolean changed = false;
    for (int i = 0; i < parts.length; i++) {
      String alias = SEGMENT_ALIASES.get(parts[i]);
      if (alias == null) {
        continue;
      }
      parts[i] = alias;
      changed = true;
    }
    return changed ? String.join(".", parts) : null;
  }

  private static ResolvedParam matchPacked(
    final String candidate,
    final WeightSchema schema,
    final Map<String, Object[]> packed
  ) {
    ResolvedParam found = null;
    for (var entry : packed.entrySet()) {
      String key = entry.getKey();
      Object[] mapping = entry.getValue();
      if (mapping == null || mapping.length < 2 || !(mapping[0] instanceof String replacement)) {
        throw new IllegalStateException("packed mapping for '" + key + "' is invalid");
      }
      String rewritten = replaceSegment(candidate, key, replacement);
      if (rewritten == null || !schema.accepts(rewritten)) {
        continue;
      }
      ResolvedParam resolved = new ResolvedParam(rewritten, mapping[1]);
      if (found != null && !found.equals(resolved)) {
        throw new IllegalStateException("ambiguous packed weight '" + candidate + "'");
      }
      found = resolved;
    }
    return found;
  }

  private static String replaceSegment(
    final String name,
    final String segment,
    final String replacement
  ) {
    String[] parts = name.split("\\.", -1);
    int found = -1;
    for (int i = 0; i < parts.length; i++) {
      if (!segment.equals(parts[i])) {
        continue;
      }
      if (found >= 0) {
        throw new IllegalStateException("ambiguous packed weight '" + name + "'");
      }
      parts[i] = replacement;
      found = i;
    }
    return found < 0 ? null : String.join(".", parts);
  }

  /**
   * Maps a language-tower or wrapper name onto the text schema. {@code model.language_model.*},
   * {@code language_model.*}, {@code text_model.*}, {@code base_model.model.*}, and a repeated
   * {@code model.model.*} become {@code model.*}. A root {@code lm_head} or {@code proj_out}
   * stays beside {@code model}. Other names gain a {@code model.} prefix when they lack one.
   */
  static String canonicalWeightName(final String weightName) {
    return canonicalFromNormalized(stripWrappers(normalizeWeightName(weightName)));
  }

  private static List<String> nameCandidates(final String weightName) {
    String normalized = normalizeWeightName(weightName);
    String stripped = stripWrappers(normalized);
    LinkedHashSet<String> names = new LinkedHashSet<>();
    names.add(canonicalFromNormalized(stripped));
    names.add(normalized);
    names.add(stripped);
    if (!stripped.startsWith("model.")) {
      names.add("model." + stripped);
    }
    addPeeledRoots(names, normalized);
    addPeeledRoots(names, stripped);
    return List.copyOf(names);
  }

  private static String canonicalFromNormalized(final String stripped) {
    String peeled = peeledRoot(stripped);
    if (peeled != null) {
      return peeled;
    }
    if (stripped.startsWith("model.") || isRootParameter(stripped)) {
      return stripped;
    }
    return "model." + stripped;
  }

  private static void addPeeledRoots(final LinkedHashSet<String> names, final String name) {
    String peeled = peeledRoot(name);
    if (peeled != null) {
      names.add(peeled);
    }
  }

  private static String peeledRoot(final String name) {
    String current = name;
    String found = null;
    while (current.startsWith("model.")) {
      current = current.substring("model.".length());
      if (isRootParameter(current)) {
        found = current;
      }
    }
    return found;
  }

  private static boolean isRootParameter(final String name) {
    return name.startsWith("lm_head.") || name.startsWith("proj_out.");
  }

  private static boolean isModuleWeight(final String paramName, final String module) {
    return paramName.equals(module + ".weight") || paramName.endsWith("." + module + ".weight");
  }

  private static String normalizeWeightName(final String weightName) {
    if (weightName == null || weightName.isBlank()) {
      throw new IllegalArgumentException("weight name must not be blank");
    }
    String name = weightName.strip().replace('/', '.');
    while (name.startsWith(".")) {
      name = name.substring(1);
    }
    while (name.contains("..")) {
      name = name.replace("..", ".");
    }
    if (name.isBlank()) {
      throw new IllegalArgumentException("weight name must not be blank");
    }
    return name;
  }

  private static String stripWrappers(final String name) {
    String current = name;
    boolean stripped = true;
    while (stripped) {
      stripped = false;
      for (String wrapper : WRAPPERS) {
        if (!current.startsWith(wrapper)) {
          continue;
        }
        current = current.substring(wrapper.length());
        stripped = true;
        break;
      }
    }
    if (current.isBlank()) {
      throw new IllegalArgumentException("weight name must not be blank");
    }
    return current;
  }

  private static Tensor alignMatrix(
    final Tensor loaded,
    final int rows,
    final int cols,
    final String label
  ) {
    requireNonNull(loaded, "tensor");
    int[] shape = loaded.shape();
    if (shape.length == 2 && shape[0] == rows && shape[1] == cols) {
      return loaded;
    }
    if (shape.length == 2 && rows != cols && shape[0] == cols && shape[1] == rows) {
      return ArchitectureFills.transposeIfMatrix(loaded);
    }
    throw new IllegalStateException(
      "weight %s has shape %s, expected [%d, %d]".formatted(
        label, Arrays.toString(shape), rows, cols));
  }

  private static void requirePositive(final int rows, final int cols, final String label) {
    if (rows <= 0 || cols <= 0) {
      throw new IllegalStateException(
        "config does not define a positive shape for " + label);
    }
  }

  private interface ShardMerge {
    void write(final Object shardId, final Tensor loaded);

    Tensor finish();
  }

  private record ResolvedParam(String paramName, Object shardId) {
  }

  private record PlannedTensor(
    SafetensorsTransport.TensorRef ref,
    String paramName,
    Object shardId
  ) {
  }

  private static final class WeightAssembler {
    private final Config.HfConfig hf;
    private final Map<String, Tensor> complete = new LinkedHashMap<>();
    private final Map<String, ShardMerge> merges = new HashMap<>();

    WeightAssembler(final Config.HfConfig hf) {
      this.hf = requireNonNull(hf, "hfConfig");
    }

    void accept(final String paramName, final Object shardId, final Tensor tensor) {
      requireNonNull(tensor, "tensor");
      if (shardId == null) {
        this.rejectDuplicate(paramName);
        this.complete.put(paramName, this.storedComplete(paramName, tensor));
        return;
      }
      if (this.complete.containsKey(paramName)) {
        throw new IllegalStateException("duplicate weight: " + paramName);
      }
      this.merges.computeIfAbsent(paramName, this::newMerge).write(shardId, tensor);
    }

    WeightBag build(final WeightSchema schema) {
      for (var entry : this.merges.entrySet()) {
        if (this.complete.containsKey(entry.getKey())) {
          throw new IllegalStateException("duplicate weight: " + entry.getKey());
        }
        this.complete.put(entry.getKey(), entry.getValue().finish());
      }
      for (String required : schema.expectedParameters()) {
        if (!this.complete.containsKey(required)) {
          throw new IllegalStateException("missing required weight after load: " + required);
        }
      }
      return new WeightBag(this.complete);
    }

    private void rejectDuplicate(final String paramName) {
      if (this.complete.containsKey(paramName) || this.merges.containsKey(paramName)) {
        throw new IllegalStateException("duplicate weight: " + paramName);
      }
    }

    private Tensor storedComplete(final String paramName, final Tensor tensor) {
      if (isModuleWeight(paramName, QKV_PROJ)) {
        int[] shape = this.qkvShape();
        return alignMatrix(tensor, shape[0], shape[1], paramName);
      }
      if (isModuleWeight(paramName, GATE_UP_PROJ)) {
        int[] shape = this.gateUpShape();
        return alignMatrix(tensor, shape[0], shape[1], paramName);
      }
      return tensor;
    }

    private ShardMerge newMerge(final String paramName) {
      if (isModuleWeight(paramName, QKV_PROJ)) {
        return new QkvMerge(this.hf, paramName);
      }
      if (isModuleWeight(paramName, GATE_UP_PROJ)) {
        return new GateUpMerge(this.hf, paramName);
      }
      throw new IllegalArgumentException("unknown packed weight target: " + paramName);
    }

    private int[] qkvShape() {
      int head = this.hf.headDim();
      int rows = Math.addExact(
        Math.multiplyExact(this.hf.numAttentionHeads(), head),
        Math.multiplyExact(Math.multiplyExact(2, this.hf.numKeyValueHeads()), head));
      int cols = this.hf.hiddenSize();
      requirePositive(rows, cols, QKV_PROJ);
      return new int[] {rows, cols};
    }

    private int[] gateUpShape() {
      int rows = Math.multiplyExact(this.hf.intermediateSize(), 2);
      int cols = this.hf.hiddenSize();
      requirePositive(rows, cols, GATE_UP_PROJ);
      return new int[] {rows, cols};
    }
  }

  private static final class QkvMerge implements ShardMerge {
    private final String paramName;
    private final Tensor weight;
    private final int headSize;
    private final int numHeads;
    private final int numKvHeads;
    private final boolean[] seen = new boolean[3];

    QkvMerge(final Config.HfConfig hf, final String paramName) {
      this.paramName = paramName;
      this.headSize = hf.headDim();
      this.numHeads = hf.numAttentionHeads();
      this.numKvHeads = hf.numKeyValueHeads();
      int rows = Math.addExact(
        Math.multiplyExact(this.numHeads, this.headSize),
        Math.multiplyExact(Math.multiplyExact(2, this.numKvHeads), this.headSize));
      requirePositive(rows, hf.hiddenSize(), paramName);
      this.weight = Tensor.zeros(rows, hf.hiddenSize());
    }

    @Override
    public void write(final Object shardId, final Tensor loaded) {
      String id = String.valueOf(shardId);
      int shardSize;
      int shardOffset;
      int seenIndex;
      switch (id) {
        case "q" -> {
          shardSize = Math.multiplyExact(this.numHeads, this.headSize);
          shardOffset = 0;
          seenIndex = 0;
        }
        case "k" -> {
          shardSize = Math.multiplyExact(this.numKvHeads, this.headSize);
          shardOffset = Math.multiplyExact(this.numHeads, this.headSize);
          seenIndex = 1;
        }
        case "v" -> {
          shardSize = Math.multiplyExact(this.numKvHeads, this.headSize);
          shardOffset = Math.addExact(
            Math.multiplyExact(this.numHeads, this.headSize),
            Math.multiplyExact(this.numKvHeads, this.headSize));
          seenIndex = 2;
        }
        default -> throw new IllegalArgumentException(
          "shard id " + shardId + " for " + this.paramName);
      }
      if (shardSize <= 0) {
        throw new IllegalStateException(
          "config does not define a positive shape for " + this.paramName);
      }
      if (this.seen[seenIndex]) {
        throw new IllegalStateException("duplicate qkv shard " + id + " for " + this.paramName);
      }
      this.copyRows(loaded, shardOffset, shardSize, this.paramName + " " + id);
      this.seen[seenIndex] = true;
    }

    private void copyRows(
      final Tensor loaded,
      final int shardOffset,
      final int shardSize,
      final String label
    ) {
      Tensor aligned = alignMatrix(loaded, shardSize, this.weight.size(1), label);
      int in = aligned.size(1);
      for (int row = 0; row < shardSize; row++) {
        System.arraycopy(
          aligned.data(), aligned.offset() + row * in,
          this.weight.data(), (shardOffset + row) * in,
          in);
      }
    }

    @Override
    public Tensor finish() {
      if (!this.seen[0] || !this.seen[1] || !this.seen[2]) {
        throw new IllegalStateException("incomplete qkv_proj merge for " + this.paramName);
      }
      return this.weight;
    }
  }

  private static final class GateUpMerge implements ShardMerge {
    private final String paramName;
    private final Tensor weight;
    private final int[] outputSizes;
    private final boolean[] seen;

    GateUpMerge(final Config.HfConfig hf, final String paramName) {
      this.paramName = paramName;
      int intermediate = hf.intermediateSize();
      requirePositive(intermediate, hf.hiddenSize(), paramName);
      this.outputSizes = new int[] {intermediate, intermediate};
      this.weight = Tensor.zeros(Math.multiplyExact(intermediate, 2), hf.hiddenSize());
      this.seen = new boolean[this.outputSizes.length];
    }

    @Override
    public void write(final Object shardId, final Tensor loaded) {
      if (!(shardId instanceof Number number)) {
        throw new IllegalArgumentException(
          "gate/up shard id must be a number for " + this.paramName + ", got " + shardId);
      }
      int id = number.intValue();
      if (id < 0 || id >= this.outputSizes.length) {
        throw new IllegalArgumentException("shard id " + shardId + " for " + this.paramName);
      }
      if (this.seen[id]) {
        throw new IllegalStateException("duplicate gate/up shard " + id + " for " + this.paramName);
      }
      int shardOffset = 0;
      for (int i = 0; i < id; i++) {
        shardOffset += this.outputSizes[i];
      }
      int shardSize = this.outputSizes[id];
      Tensor aligned =
        alignMatrix(loaded, shardSize, this.weight.size(1), this.paramName + " " + id);
      int in = aligned.size(1);
      for (int row = 0; row < shardSize; row++) {
        System.arraycopy(
          aligned.data(), aligned.offset() + row * in,
          this.weight.data(), (shardOffset + row) * in,
          in);
      }
      this.seen[id] = true;
    }

    @Override
    public Tensor finish() {
      for (boolean ok : this.seen) {
        if (!ok) {
          throw new IllegalStateException("incomplete gate_up_proj merge for " + this.paramName);
        }
      }
      return this.weight;
    }
  }
}
