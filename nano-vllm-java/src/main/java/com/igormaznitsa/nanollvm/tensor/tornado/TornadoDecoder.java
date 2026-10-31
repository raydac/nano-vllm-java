package com.igormaznitsa.nanollvm.tensor.tornado;

import static com.igormaznitsa.nanollvm.layers.DeviceDecoders.Activation.GELU;
import static com.igormaznitsa.nanollvm.layers.DeviceDecoders.Shape.HIGHWAY;
import static com.igormaznitsa.nanollvm.layers.DeviceDecoders.Shape.NORM_BRANCH;
import static com.igormaznitsa.nanollvm.layers.DeviceDecoders.Shape.PLAIN_BRANCH;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.CACHE_OFFSET;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.HAS_RESIDUAL;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.KEY_LENGTH;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.POSITION;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.SLOT;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoQwen3LayerKernels.WINDOW;
import static java.util.Objects.requireNonNull;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.DeviceDecoders.Gain;
import com.igormaznitsa.nanollvm.layers.DeviceDecoders.Layer;
import com.igormaznitsa.nanollvm.layers.DeviceDecoders.Matrix;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * Decoder layers on TornadoVM. A frame runs four layers in one task graph so a token waits for
 * the device once per group. Tornado encodes a graph into a fixed 8192-byte buffer, which is why
 * the group stays at four and not the whole stack. Prefill walks that graph once per token.
 */
public final class TornadoDecoder {

  private static final int LAYERS_PER_GRAPH = 4;

  private static final Map<Key, LayerPlan> PLANS = new HashMap<>();

  private static final Map<GroupKey, LayerPlan.GroupPlan> GROUPS = new HashMap<>();

  private static final ThreadLocal<Frame> FRAME = new ThreadLocal<>();

  private TornadoDecoder() {
  }

  public static Tensor[] forward(final Layer layer) {
    requireNonNull(layer, "layer");
    Frame frame = FRAME.get();
    if (!accepted(layer)) {
      if (frame != null) {
        frame.flush();
      }
      return null;
    }
    if (frame == null) {
      return runImmediate(layer);
    }
    return frame.record(layer);
  }

  public static void openFrame() {
    Frame current = FRAME.get();
    if (current != null) {
      current.flush();
    }
    FRAME.set(new Frame());
  }

  public static void closeFrame() {
    Frame current = FRAME.get();
    FRAME.remove();
    if (current != null) {
      current.flush();
    }
  }

  private static Tensor[] runImmediate(final Layer layer) {
    int rows = layer.hidden().size(0);
    int hidden = layer.hidden().size(1);
    Tensor hiddenOut = Tensor.zeros(rows, hidden);
    Tensor residualOut = layer.shape() == HIGHWAY ? Tensor.zeros(rows, hidden) : null;
    LayerPlan plan = plan(layer);
    for (int token = 0; token < rows; token++) {
      plan.token(
        layer, token, hiddenOut.data(), residualOut == null ? null : residualOut.data());
    }
    return residualOut == null
      ? new Tensor[] {hiddenOut}
      : new Tensor[] {hiddenOut, residualOut};
  }

  static void releaseWeights(final float[]... weights) {
    if (weights == null || weights.length == 0) {
      return;
    }
    Set<float[]> drop = Collections.newSetFromMap(new IdentityHashMap<>());
    for (float[] weight : weights) {
      if (weight != null) {
        drop.add(weight);
      }
    }
    TornadoLaunchLock.lock();
    try {
      PLANS.entrySet().removeIf(entry -> {
        if (!entry.getValue().uses(drop)) {
          return false;
        }
        entry.getValue().closeQuietly();
        return true;
      });
      GROUPS.entrySet().removeIf(entry -> {
        if (!entry.getValue().uses(drop)) {
          return false;
        }
        entry.getValue().closeQuietly();
        return true;
      });
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static boolean accepted(final Layer layer) {
    if (layer.hidden().shape().length != 2 || layer.hidden().size(0) <= 0) {
      return false;
    }
    if (!layer.attention()) {
      return layer.shape() != HIGHWAY || layer.residual() != null;
    }
    Context context = layer.context();
    int rows = layer.hidden().size(0);
    if (context == null || layer.rope() == null || layer.positions() == null
      || layer.keyCache() == null || layer.valueCache() == null
      || context.slotMapping() == null || context.slotMapping().length < rows) {
      return false;
    }
    if (context.isPrefill()) {
      int[] queries = context.cuSeqlensQ();
      return queries != null && queries.length == 2 && prefixSlotsReady(context);
    }
    return context.contextLens() != null && context.contextLens().length == 1
      && context.pagedSlots() != null && context.pagedSlots()[0] != null;
  }

  private static boolean prefixSlotsReady(final Context context) {
    int[] queries = context.cuSeqlensQ();
    int[] keys = context.cuSeqlensK();
    int start = keys == null || queries == null ? 0 : keys[1] - queries[1];
    if (start <= 0) {
      return true;
    }
    return context.pagedSlots() != null && context.pagedSlots()[0] != null;
  }

  private static LayerPlan plan(final Layer layer) {
    TornadoLaunchLock.lock();
    try {
      Key key = new Key(primary(layer), layer.keyCache() == null ? null : layer.keyCache().data());
      LayerPlan cached = PLANS.get(key);
      if (cached != null) {
        return cached;
      }
      LayerPlan compiled = LayerPlan.compile(layer);
      PLANS.put(key, compiled);
      return compiled;
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static float[] primary(final Layer layer) {
    if (layer.qkv() != null) {
      return layer.qkv().data();
    }
    if (layer.query() != null) {
      return layer.query().data();
    }
    if (layer.gateUp() != null) {
      return layer.gateUp().data();
    }
    if (layer.gate() != null) {
      return layer.gate().data();
    }
    if (layer.down() == null) {
      throw new IllegalArgumentException("device layer has no projection");
    }
    return layer.down().data();
  }

  private record Key(float[] weight, float[] keyCache) {
    @Override
    public boolean equals(final Object other) {
      return other instanceof Key(float[] weight1, float[] cache)
        && this.weight == weight1 && this.keyCache == cache;
    }

    @Override
    public int hashCode() {
      return System.identityHashCode(this.weight) * 31 + System.identityHashCode(this.keyCache);
    }
  }

  private static Tensor[] layerOutputs(final Layer layer) {
    int rows = layer.hidden().size(0);
    int hidden = layer.hidden().size(1);
    if (layer.shape() == HIGHWAY) {
      return new Tensor[] {Tensor.zeros(rows, hidden), Tensor.zeros(rows, hidden)};
    }
    return new Tensor[] {Tensor.zeros(rows, hidden)};
  }

  private static LayerPlan.GroupPlan group(final List<Pending> batch) {
    GroupKey key = GroupKey.of(batch);
    TornadoLaunchLock.lock();
    try {
      LayerPlan.GroupPlan cached = GROUPS.get(key);
      if (cached != null) {
        return cached;
      }
      LayerPlan.GroupPlan compiled = LayerPlan.compileGroup(key, batch);
      GROUPS.put(key, compiled);
      return compiled;
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private record Pending(Layer layer, Tensor[] outputs) {
  }

  private static final class Frame {
    private final List<Pending> pending = new ArrayList<>();

    private Tensor[] record(final Layer layer) {
      if (!this.pending.isEmpty()
        && this.pending.getFirst().layer.hidden().size(0) != layer.hidden().size(0)) {
        this.flush();
      }
      Tensor[] outputs = layerOutputs(layer);
      this.pending.add(new Pending(layer, outputs));
      if (this.pending.size() == LAYERS_PER_GRAPH) {
        this.flush();
      }
      return outputs;
    }

    private void flush() {
      if (this.pending.isEmpty()) {
        return;
      }
      List<Pending> batch = List.copyOf(this.pending);
      this.pending.clear();
      group(batch).run(batch);
    }
  }

  private static final class GroupKey {
    private final float[][] weights;
    private final float[][] caches;
    private final int hash;

    private GroupKey(final float[][] weights, final float[][] caches) {
      this.weights = weights;
      this.caches = caches;
      int mixed = weights.length;
      for (int index = 0; index < weights.length; index++) {
        mixed = mixed * 31 + System.identityHashCode(weights[index]);
        mixed = mixed * 31 + System.identityHashCode(caches[index]);
      }
      this.hash = mixed;
    }

    private static GroupKey of(final List<Pending> batch) {
      float[][] weights = new float[batch.size()][];
      float[][] caches = new float[batch.size()][];
      for (int index = 0; index < batch.size(); index++) {
        Layer layer = batch.get(index).layer;
        weights[index] = primary(layer);
        caches[index] = layer.keyCache() == null ? null : layer.keyCache().data();
      }
      return new GroupKey(weights, caches);
    }

    @Override
    public boolean equals(final Object other) {
      if (!(other instanceof GroupKey that) || this.weights.length != that.weights.length) {
        return false;
      }
      for (int index = 0; index < this.weights.length; index++) {
        if (this.weights[index] != that.weights[index] ||
          this.caches[index] != that.caches[index]) {
          return false;
        }
      }
      return true;
    }

    @Override
    public int hashCode() {
      return this.hash;
    }
  }

  private static final class LayerPlan {
    private final TornadoExecutionPlan plan;
    private final float[][] sources;
    private final float[] x;
    private final float[] residual;
    private final float[] angles;
    private final float[] perLayer;
    private final float[] hiddenOut;
    private final float[] residualOut;
    private final float[] rotatedKey;
    private final float[] storedValue;
    private final float[] keyCache;
    private final float[] valueCache;
    private final int[] params;
    private final int[] slots;
    private final int hidden;
    private final int keySize;
    private final int cacheOffset;
    private final int valueOffset;
    private final int window;
    private final boolean attention;
    private final boolean writeKv;
    private final RotaryEmbedding rope;
    private int lastKeyLength = -1;

    private LayerPlan(
      final TornadoExecutionPlan plan,
      final float[][] sources,
      final float[] x, final float[] residual, final float[] angles, final float[] perLayer,
      final float[] hiddenOut, final float[] residualOut,
      final float[] rotatedKey, final float[] storedValue,
      final float[] keyCache, final float[] valueCache,
      final int[] params, final int[] slots,
      final int hidden, final int keySize, final int cacheOffset, final int valueOffset,
      final int window, final boolean attention, final boolean writeKv,
      final RotaryEmbedding rope
    ) {
      this.plan = plan;
      this.sources = sources;
      this.x = x;
      this.residual = residual;
      this.angles = angles;
      this.perLayer = perLayer;
      this.hiddenOut = hiddenOut;
      this.residualOut = residualOut;
      this.rotatedKey = rotatedKey;
      this.storedValue = storedValue;
      this.keyCache = keyCache;
      this.valueCache = valueCache;
      this.params = params;
      this.slots = slots;
      this.hidden = hidden;
      this.keySize = keySize;
      this.cacheOffset = cacheOffset;
      this.valueOffset = valueOffset;
      this.window = window;
      this.attention = attention;
      this.writeKv = writeKv;
      this.rope = rope;
    }

    static LayerPlan compile(final Layer layer) {
      int hidden = layer.hidden().size(1);
      Draft draft = new Draft("nanollvm-layer-" + System.identityHashCode(primary(layer)));
      float[] x = new float[hidden];
      float[] residual = new float[hidden];
      draft.every(x);
      draft.every(residual);
      List<float[]> host = new ArrayList<>();
      Stage stage = append(draft, layer, x, residual, true, host);
      draft.host(host);
      return stage.plan(draft.open(), sources(layer));
    }

    static GroupPlan compileGroup(final GroupKey key, final List<Pending> batch) {
      Layer first = batch.getFirst().layer;
      int hidden = first.hidden().size(1);
      Draft draft = new Draft("nanollvm-group-" + Integer.toUnsignedString(key.hashCode()));
      float[] input = new float[hidden];
      float[] residual = new float[hidden];
      draft.every(input);
      draft.every(residual);
      List<float[]> host = new ArrayList<>();
      List<Stage> stages = new ArrayList<>();
      List<float[]> weightSources = new ArrayList<>();
      float[] layerInput = input;
      float[] layerResidual = residual;
      for (int index = 0; index < batch.size(); index++) {
        Layer layer = batch.get(index).layer;
        draft.setPrefix("l" + index + "_");
        Stage stage = append(draft, layer, layerInput, layerResidual, true, host);
        stages.add(stage);
        Collections.addAll(weightSources, sources(layer));
        layerInput = stage.hiddenOut;
        layerResidual = stage.residualOut == null ? layerResidual : stage.residualOut;
      }
      draft.host(host);
      return new GroupPlan(draft.open(), stages, weightSources.toArray(float[][]::new), hidden);
    }

    private static Stage append(
      final Draft draft, final Layer layer,
      final float[] x, final float[] residual,
      final boolean publishActivation, final List<float[]> host
    ) {
      int hidden = x.length;
      float[] epsilon = {layer.epsilon()};
      draft.resident(epsilon);

      float[] angles = null;
      int[] params = null;
      int[] slots = null;
      float[] rotatedKey = null;
      float[] storedValue = null;
      float[] keyCache = null;
      float[] valueCache = null;
      int keySize = 0;
      int cacheOffset = 0;
      int valueOffset = 0;
      float[] mlpInput;
      float[] carried;

      if (!layer.attention()) {
        mlpInput = mixTail(draft, layer, x, residual, epsilon, hidden);
        carried = layer.shape() == HIGHWAY ? draft.carried : x;
      } else {
        AttentionBuffers attention = projectAttention(
          draft, layer, x, residual, epsilon, hidden);
        angles = attention.angles;
        params = attention.params;
        slots = attention.slots;
        rotatedKey = attention.rotatedKey;
        storedValue = attention.storedValue;
        keyCache = attention.keyCache;
        valueCache = attention.valueCache;
        keySize = attention.keySize;
        cacheOffset = attention.cacheOffset;
        valueOffset = attention.valueOffset;
        carried = attention.carried;
        mlpInput = attention.mlpInput;
      }

      float[] result = projectMlp(draft, layer, mlpInput, carried, epsilon, hidden);
      result = projectPerLayer(draft, layer, result, epsilon, hidden);
      if (publishActivation) {
        host.add(result);
        if (layer.shape() == HIGHWAY) {
          host.add(carried);
        }
      }
      if (layer.writeKv()) {
        host.add(rotatedKey);
        host.add(storedValue);
      }
      return new Stage(
        x, residual, angles, draft.perLayerRow, result, carried,
        rotatedKey, storedValue, keyCache, valueCache, params, slots,
        hidden, keySize, cacheOffset, valueOffset, layer.window(),
        layer.attention(), layer.writeKv(), layer.rope());
    }

    private static float[] mixTail(
      final Draft draft, final Layer layer,
      final float[] x, final float[] residual, final float[] epsilon, final int hidden
    ) {
      if (layer.shape() == NORM_BRANCH) {
        throw new IllegalArgumentException("norm-branch mlp tail is not a device graph");
      }
      float[] post = new float[hidden];
      draft.resident(post);
      if (layer.shape() == HIGHWAY) {
        float[] carried = new float[hidden];
        float[] gain = requireGain(layer.postAttentionNorm(), hidden, "post-attention");
        draft.resident(carried);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("post"), TornadoNormKernels::addRmsNormRows,
          x, residual, carried, gain, post, epsilon, 1, hidden);
        draft.carried = carried;
        return post;
      }
      float[] gain = requireGain(layer.preMlpNorm(), hidden, "pre-mlp");
      draft.resident(gain);
      draft.graph =
        draft.graph.task(draft.id("pre"), TornadoNormKernels::rmsNormRows, x, gain, post, epsilon,
          1, hidden);
      return post;
    }

    private static AttentionBuffers projectAttention(
      final Draft draft, final Layer layer,
      final float[] x, final float[] residual, final float[] epsilon, final int hidden
    ) {
      int querySize = Math.multiplyExact(layer.numHeads(), layer.headDim());
      int keySize = Math.multiplyExact(layer.numKvHeads(), layer.headDim());
      if ((layer.rope().getRotaryDim() & 1) != 0 || layer.numHeads() % layer.numKvHeads() != 0) {
        throw new IllegalArgumentException("device attention needs even rotary groups");
      }
      if (layer.keyCache().offset() != layer.valueCache().offset()) {
        throw new IllegalArgumentException("key and value caches start at different offsets");
      }
      int slotCount = layer.keyCache().numel() / keySize;
      if (slotCount <= 0 || layer.valueCache().numel() / keySize != slotCount) {
        throw new IllegalArgumentException("key and value caches differ");
      }

      float[] stream = new float[hidden];
      float[] normed = new float[hidden];
      float[] gainIn = requireGain(layer.inputNorm(), hidden, "input");
      int[] params = new int[6];
      draft.resident(stream);
      draft.resident(normed);
      draft.resident(gainIn);
      draft.every(params);
      draft.graph = draft.graph.task(draft.id("norm"), TornadoQwen3LayerKernels::residualRms,
        x, residual, stream, normed, gainIn, epsilon, params);

      float[] query = new float[querySize];
      float[] key = new float[keySize];
      float[] value = new float[keySize];
      draft.resident(query);
      draft.resident(key);
      draft.resident(value);
      float[] queryGate =
        projectQueries(draft, layer, normed, query, key, value, querySize, keySize);

      float[] queryForRope = query;
      float[] keyForRope = key;
      if (layer.queryNorm() != null) {
        queryForRope = new float[querySize];
        float[] gain = bake(layer.queryNorm(), layer.headDim());
        draft.resident(queryForRope);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("qnorm"), TornadoNormKernels::rmsNormRows,
          query, gain, queryForRope, epsilon, layer.numHeads(), layer.headDim());
      }
      boolean projectsKeys = layer.qkv() != null || layer.key() != null;
      if (layer.keyNorm() != null && projectsKeys) {
        keyForRope = new float[keySize];
        float[] gain = bake(layer.keyNorm(), layer.headDim());
        draft.resident(keyForRope);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("knorm"), TornadoNormKernels::rmsNormRows,
          key, gain, keyForRope, epsilon, layer.numKvHeads(), layer.headDim());
      }
      float[] storedValue = value;
      if (layer.valueNorm() != null && layer.value() != null) {
        storedValue = new float[keySize];
        float[] gain = bake(layer.valueNorm(), layer.headDim());
        draft.resident(storedValue);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("vnorm"), TornadoNormKernels::rmsNormRows,
          value, gain, storedValue, epsilon, layer.numKvHeads(), layer.headDim());
      }

      float[] angles = new float[layer.rope().getRotaryDim()];
      float[] rotatedQuery = new float[querySize];
      draft.every(angles);
      draft.resident(rotatedQuery);
      draft.graph = draft.graph.task(draft.id("ropeq"), TornadoQwen3LayerKernels::ropeHeads,
        queryForRope, rotatedQuery, angles, layer.numHeads(), layer.headDim());
      float[] rotatedKey = null;
      if (projectsKeys) {
        rotatedKey = new float[keySize];
        draft.resident(rotatedKey);
        draft.graph = draft.graph.task(draft.id("ropek"), TornadoQwen3LayerKernels::ropeHeads,
          keyForRope, rotatedKey, angles, layer.numKvHeads(), layer.headDim());
      }

      float[] cacheKey = layer.keyCache().data();
      float[] cacheValue = layer.valueCache().data();
      if (layer.writeKv()) {
        draft.resident(cacheKey);
        draft.resident(cacheValue);
        draft.graph = draft.graph.task(draft.id("store"), TornadoQwen3LayerKernels::storeKv,
          rotatedKey, storedValue, cacheKey, cacheValue, params);
      } else {
        draft.every(cacheKey);
        draft.every(cacheValue);
      }
      int[] slots = new int[slotCount];
      float[] coeff = {layer.scoreScale(), Float.NEGATIVE_INFINITY};
      float[] attended = new float[querySize];
      draft.every(slots);
      draft.resident(coeff);
      draft.resident(attended);
      draft.graph = draft.graph.task(draft.id("attend"), TornadoQwen3LayerKernels::attendOne,
        rotatedQuery, cacheKey, cacheValue, slots, attended, coeff, params,
        layer.numHeads(), layer.numKvHeads(), layer.headDim());

      float[] mixed = attended;
      if (queryGate != null) {
        mixed = new float[querySize];
        draft.resident(mixed);
        draft.graph =
          draft.graph.task(draft.id("qgate"), TornadoQwen3LayerKernels::sigmoidMul, attended,
            queryGate,
            mixed);
      }
      float[] projected = new float[hidden];
      draft.resident(projected);
      requireMatrix(layer.output(), hidden, querySize, "output");
      draft.gemv("oproj", mixed, layer.output(), projected);

      float[] mlpInput = combineAttention(draft, layer, stream, projected, epsilon, hidden);
      AttentionBuffers buffers = new AttentionBuffers();
      buffers.angles = angles;
      buffers.params = params;
      buffers.slots = slots;
      buffers.rotatedKey = rotatedKey;
      buffers.storedValue = storedValue;
      buffers.keyCache = cacheKey;
      buffers.valueCache = cacheValue;
      buffers.keySize = keySize;
      buffers.cacheOffset = layer.keyCache().offset();
      buffers.valueOffset = layer.valueCache().offset();
      buffers.carried = draft.carried;
      buffers.mlpInput = mlpInput;
      return buffers;
    }

    private static float[] projectQueries(
      final Draft draft, final Layer layer, final float[] normed,
      final float[] query, final float[] key, final float[] value,
      final int querySize, final int keySize
    ) {
      int hidden = normed.length;
      if (layer.qkv() != null) {
        requireMatrix(layer.qkv(), querySize + keySize * 2, hidden, "qkv");
        float[] qkvOut = new float[querySize + keySize * 2];
        draft.resident(qkvOut);
        draft.gemv("qkv", normed, layer.qkv(), qkvOut);
        draft.graph =
          draft.graph.task(draft.id("split"), TornadoQwen3LayerKernels::splitQkv, qkvOut, query,
            key, value);
        return null;
      }
      if (layer.query() == null) {
        throw new IllegalArgumentException("device attention needs a query projection");
      }
      float[] queryGate = null;
      if (layer.queryGate()) {
        requireMatrix(layer.query(), querySize * 2, hidden, "query");
        float[] packed = new float[querySize * 2];
        queryGate = new float[querySize];
        draft.resident(packed);
        draft.resident(queryGate);
        draft.gemv("q", normed, layer.query(), packed);
        draft.graph =
          draft.graph.task(draft.id("qsplit"), TornadoQwen3LayerKernels::splitHalves, packed, query,
            queryGate);
      } else {
        requireMatrix(layer.query(), querySize, hidden, "query");
        draft.gemv("q", normed, layer.query(), query);
      }
      if (layer.writeKv() || layer.key() != null) {
        requireMatrix(layer.key(), keySize, hidden, "key");
        requireMatrix(layer.value(), keySize, hidden, "value");
        draft.gemv("k", normed, layer.key(), key);
        draft.gemv("v", normed, layer.value(), value);
      }
      return queryGate;
    }

    private static float[] combineAttention(
      final Draft draft, final Layer layer,
      final float[] stream, final float[] projected, final float[] epsilon, final int hidden
    ) {
      if (layer.shape() == HIGHWAY) {
        float[] carried = new float[hidden];
        float[] post = new float[hidden];
        float[] gain = requireGain(layer.postAttentionNorm(), hidden, "post-attention");
        draft.resident(carried);
        draft.resident(post);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("post"), TornadoNormKernels::addRmsNormRows,
          projected, stream, carried, gain, post, epsilon, 1, hidden);
        draft.carried = carried;
        return post;
      }
      float[] carried = new float[hidden];
      draft.resident(carried);
      if (layer.shape() == NORM_BRANCH) {
        float[] branch = new float[hidden];
        float[] gain = requireGain(layer.postAttentionNorm(), hidden, "post-attention");
        draft.resident(branch);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("post"), TornadoNormKernels::rmsNormRows,
          projected, gain, branch, epsilon, 1, hidden);
        draft.graph =
          draft.graph.task(draft.id("addAttn"), TornadoElementKernels::add, stream, branch, carried,
            hidden);
      } else if (layer.shape() == PLAIN_BRANCH) {
        draft.graph =
          draft.graph.task(draft.id("addAttn"), TornadoElementKernels::add, stream, projected,
            carried,
            hidden);
      }
      float[] post = new float[hidden];
      float[] gain = requireGain(layer.preMlpNorm(), hidden, "pre-mlp");
      draft.resident(post);
      draft.resident(gain);
      draft.graph =
        draft.graph.task(draft.id("pre"), TornadoNormKernels::rmsNormRows, carried, gain, post,
          epsilon, 1,
          hidden);
      draft.carried = carried;
      return post;
    }

    private static float[] projectMlp(
      final Draft draft, final Layer layer,
      final float[] mlpInput, final float[] carried, final float[] epsilon, final int hidden
    ) {
      int intermediate = intermediate(layer, hidden);
      float[] activated = new float[intermediate];
      draft.resident(activated);
      if (layer.gateUp() != null) {
        requireMatrix(layer.gateUp(), intermediate * 2, hidden, "gate");
        float[] gateOut = new float[intermediate * 2];
        draft.resident(gateOut);
        draft.gemv("gate", mlpInput, layer.gateUp(), gateOut);
        if (layer.activation() == GELU) {
          draft.graph =
            draft.graph.task(draft.id("act"), TornadoQwen3LayerKernels::geluHalves, gateOut,
              activated);
        } else {
          draft.graph =
            draft.graph.task(draft.id("act"), TornadoQwen3LayerKernels::siluHalves, gateOut,
              activated);
        }
      } else {
        requireMatrix(layer.gate(), intermediate, hidden, "gate");
        requireMatrix(layer.up(), intermediate, hidden, "up");
        float[] gateOut = new float[intermediate];
        float[] upOut = new float[intermediate];
        draft.resident(gateOut);
        draft.resident(upOut);
        draft.gemv("gate", mlpInput, layer.gate(), gateOut);
        draft.gemv("up", mlpInput, layer.up(), upOut);
        if (layer.activation() == GELU) {
          draft.graph =
            draft.graph.task(draft.id("act"), TornadoElementKernels::geluTanhMul, gateOut, upOut,
              activated,
              intermediate);
        } else {
          draft.graph =
            draft.graph.task(draft.id("act"), TornadoQwen3LayerKernels::siluMul, gateOut, upOut,
              activated);
        }
      }
      boolean direct = layer.shape() == HIGHWAY && layer.perLayerGate() == null;
      float[] mlpOut = new float[hidden];
      draft.resident(mlpOut);
      requireMatrix(layer.down(), hidden, intermediate, "down");
      draft.gemv("down", activated, layer.down(), mlpOut);
      if (direct) {
        return mlpOut;
      }
      float[] hiddenOut = new float[hidden];
      draft.resident(hiddenOut);
      if (layer.shape() == NORM_BRANCH) {
        float[] normed = new float[hidden];
        float[] gain = requireGain(layer.postMlpNorm(), hidden, "post-mlp");
        draft.resident(normed);
        draft.resident(gain);
        draft.graph = draft.graph.task(draft.id("postMlp"), TornadoNormKernels::rmsNormRows,
          mlpOut, gain, normed, epsilon, 1, hidden);
        draft.graph =
          draft.graph.task(draft.id("addMlp"), TornadoElementKernels::add, carried, normed,
            hiddenOut,
            hidden);
      } else {
        draft.graph =
          draft.graph.task(draft.id("addMlp"), TornadoElementKernels::add, carried, mlpOut,
            hiddenOut,
            hidden);
      }
      return hiddenOut;
    }

    private static float[] projectPerLayer(
      final Draft draft, final Layer layer, final float[] hiddenState,
      final float[] epsilon, final int hidden
    ) {
      if (layer.perLayerGate() == null) {
        return hiddenState;
      }
      int width = layer.perLayerGate().rows();
      requireMatrix(layer.perLayerGate(), width, hidden, "per-layer gate");
      requireMatrix(layer.perLayerProjection(), hidden, width, "per-layer projection");
      if (layer.perLayerInput() == null || layer.perLayerInput().size(1) != width) {
        throw new IllegalArgumentException("per-layer input width does not match its gate");
      }
      float[] row = new float[width];
      float[] gated = new float[width];
      float[] projected = new float[hidden];
      float[] normed = new float[hidden];
      float[] summed = new float[hidden];
      float[] scaled = new float[hidden];
      float[] gain = requireGain(layer.postPerLayerNorm(), hidden, "post-per-layer");
      float[] factor = {layer.layerScale()};
      draft.every(row);
      draft.resident(gated);
      draft.resident(projected);
      draft.resident(normed);
      draft.resident(summed);
      draft.resident(scaled);
      draft.resident(gain);
      draft.resident(factor);
      draft.perLayerRow = row;
      float[] gate = new float[width];
      draft.resident(gate);
      draft.gemv("ple", hiddenState, layer.perLayerGate(), gate);
      draft.graph =
        draft.graph.task(draft.id("pleAct"), TornadoElementKernels::geluTanhMul, gate, row, gated,
          width);
      draft.gemv("pleProj", gated, layer.perLayerProjection(), projected);
      draft.graph = draft.graph.task(draft.id("pleNorm"), TornadoNormKernels::rmsNormRows,
        projected, gain, normed, epsilon, 1, hidden);
      draft.graph =
        draft.graph.task(draft.id("pleAdd"), TornadoElementKernels::add, hiddenState, normed,
          summed, hidden);
      draft.graph =
        draft.graph.task(draft.id("pleScale"), TornadoElementKernels::scale, summed, factor, scaled,
          hidden);
      return scaled;
    }

    void token(
      final Layer layer, final int token, final float[] hiddenDest, final float[] residualDest
    ) {
      this.loadInput(layer, token);
      int keyLength = this.prepare(layer, token);
      TornadoLaunchLock.lock();
      try {
        if (this.needsCacheRefresh(keyLength)) {
          this.plan.transferToDevice(this.keyCache, this.valueCache);
        }
        this.plan.execute();
        this.rememberKeyLength(keyLength);
      } finally {
        TornadoLaunchLock.unlock();
      }
      this.publish(hiddenDest, residualDest, token * this.hidden);
    }

    private static int intermediate(final Layer layer, final int hidden) {
      if (layer.gateUp() != null) {
        if ((layer.gateUp().rows() & 1) != 0 || layer.gateUp().columns() != hidden) {
          throw new IllegalArgumentException(
            "fused gate projection is not [2 * intermediate, hidden]");
        }
        return layer.gateUp().rows() / 2;
      }
      if (layer.gate() == null || layer.gate().columns() != hidden) {
        throw new IllegalArgumentException("device mlp needs a gate projection");
      }
      return layer.gate().rows();
    }

    private static int keyLength(final Context context, final int token) {
      if (!context.isPrefill()) {
        return context.contextLens()[0];
      }
      int[] queries = context.cuSeqlensQ();
      int[] keys = context.cuSeqlensK();
      int start = keys == null ? 0 : keys[1] - queries[1];
      return start + token + 1;
    }

    private static int[] sourceSlots(final Context context, final int keyLength) {
      int[] source = context.pagedSlots() != null && context.pagedSlots()[0] != null
        ? context.pagedSlots()[0]
        : context.slotMapping();
      if (source == null || source.length < keyLength) {
        throw new IllegalArgumentException("key slots shorter than the visible context");
      }
      return source;
    }

    private static void requireMatrix(
      final Matrix weight, final int rows, final int columns, final String name
    ) {
      if (weight == null || weight.rows() != rows || weight.columns() != columns) {
        throw new IllegalArgumentException(
          "%s weight is missing or not [%d, %d]".formatted(name, rows, columns));
      }
      if (weight.bias() != null && weight.bias().length < rows) {
        throw new IllegalArgumentException(name + " bias is shorter than the projection");
      }
    }

    private static float[] requireGain(final Gain gain, final int width, final String name) {
      if (gain == null) {
        throw new IllegalArgumentException(name + " norm is missing");
      }
      return bake(gain, width);
    }

    private static float[] bake(final Gain gain, final int width) {
      float[] baked = new float[width];
      if (gain.data() == null) {
        Arrays.fill(baked, 1f);
        return baked;
      }
      if (gain.width() != width) {
        throw new IllegalArgumentException(
          "norm width %d, expected %d".formatted(gain.width(), width));
      }
      for (int i = 0; i < width; i++) {
        float value = gain.data()[gain.offset() + i];
        baked[i] = gain.onePlus() ? 1f + value : value;
      }
      return baked;
    }

    private static float[][] sources(final Layer layer) {
      List<float[]> found = new ArrayList<>();
      add(found, layer.qkv());
      add(found, layer.query());
      add(found, layer.key());
      add(found, layer.value());
      add(found, layer.output());
      add(found, layer.gateUp());
      add(found, layer.gate());
      add(found, layer.up());
      add(found, layer.down());
      add(found, layer.perLayerGate());
      add(found, layer.perLayerProjection());
      add(found, layer.inputNorm());
      add(found, layer.postAttentionNorm());
      add(found, layer.preMlpNorm());
      add(found, layer.postMlpNorm());
      add(found, layer.queryNorm());
      add(found, layer.keyNorm());
      add(found, layer.valueNorm());
      add(found, layer.postPerLayerNorm());
      return found.toArray(float[][]::new);
    }

    private static void add(final List<float[]> found, final Matrix matrix) {
      if (matrix == null) {
        return;
      }
      found.add(matrix.data());
      if (matrix.bias() != null) {
        found.add(matrix.bias());
      }
    }

    private static void add(final List<float[]> found, final Gain gain) {
      if (gain != null && gain.data() != null) {
        found.add(gain.data());
      }
    }

    private void loadInput(final Layer layer, final int token) {
      int row = token * this.hidden;
      System.arraycopy(
        layer.hidden().data(), layer.hidden().offset() + row, this.x, 0, this.hidden);
      if (layer.residual() != null) {
        System.arraycopy(
          layer.residual().data(), layer.residual().offset() + row, this.residual, 0, this.hidden);
      }
    }

    private int prepare(final Layer layer, final int token) {
      if (this.perLayer != null) {
        System.arraycopy(
          layer.perLayerInput().data(),
          layer.perLayerInput().offset() + token * this.perLayer.length,
          this.perLayer, 0, this.perLayer.length);
      }
      if (!this.attention) {
        return 0;
      }
      int position = Math.round(layer.positions().get(token));
      this.rope.copyAngles(position, this.angles);
      int keyLength = keyLength(layer.context(), token);
      int[] source = sourceSlots(layer.context(), keyLength);
      int slot = layer.context().slotMapping()[token];
      if (keyLength > this.slots.length || slot < 0) {
        throw new IllegalArgumentException("device cache cannot hold this token");
      }
      System.arraycopy(source, 0, this.slots, 0, keyLength);
      this.params[POSITION] = position;
      this.params[SLOT] = slot;
      this.params[KEY_LENGTH] = keyLength;
      this.params[HAS_RESIDUAL] = layer.shape() == HIGHWAY && layer.residual() != null ? 1 : 0;
      this.params[WINDOW] = this.window;
      this.params[CACHE_OFFSET] = this.cacheOffset;
      return keyLength;
    }

    private boolean needsCacheRefresh(final int keyLength) {
      return this.writeKv && this.lastKeyLength >= 0 && keyLength <= this.lastKeyLength;
    }

    private void rememberKeyLength(final int keyLength) {
      if (this.writeKv) {
        this.lastKeyLength = keyLength;
      }
    }

    private void publish(final float[] hiddenDest, final float[] residualDest, final int row) {
      System.arraycopy(this.hiddenOut, 0, hiddenDest, row, this.hidden);
      if (residualDest != null) {
        System.arraycopy(this.residualOut, 0, residualDest, row, this.hidden);
      }
      if (!this.writeKv) {
        return;
      }
      int slot = this.params[SLOT];
      int keyBase = this.cacheOffset + slot * this.keySize;
      int valueBase = this.valueOffset + slot * this.keySize;
      System.arraycopy(this.rotatedKey, 0, this.keyCache, keyBase, this.keySize);
      System.arraycopy(this.storedValue, 0, this.valueCache, valueBase, this.keySize);
    }

    private static final class Stage {
      private final float[] x;
      private final float[] residual;
      private final float[] angles;
      private final float[] perLayer;
      private final float[] hiddenOut;
      private final float[] residualOut;
      private final float[] rotatedKey;
      private final float[] storedValue;
      private final float[] keyCache;
      private final float[] valueCache;
      private final int[] params;
      private final int[] slots;
      private final int hidden;
      private final int keySize;
      private final int cacheOffset;
      private final int valueOffset;
      private final int window;
      private final boolean attention;
      private final boolean writeKv;
      private final RotaryEmbedding rope;

      private Stage(
        final float[] x, final float[] residual, final float[] angles, final float[] perLayer,
        final float[] hiddenOut, final float[] residualOut,
        final float[] rotatedKey, final float[] storedValue,
        final float[] keyCache, final float[] valueCache,
        final int[] params, final int[] slots,
        final int hidden, final int keySize, final int cacheOffset, final int valueOffset,
        final int window, final boolean attention, final boolean writeKv,
        final RotaryEmbedding rope
      ) {
        this.x = x;
        this.residual = residual;
        this.angles = angles;
        this.perLayer = perLayer;
        this.hiddenOut = hiddenOut;
        this.residualOut = residualOut;
        this.rotatedKey = rotatedKey;
        this.storedValue = storedValue;
        this.keyCache = keyCache;
        this.valueCache = valueCache;
        this.params = params;
        this.slots = slots;
        this.hidden = hidden;
        this.keySize = keySize;
        this.cacheOffset = cacheOffset;
        this.valueOffset = valueOffset;
        this.window = window;
        this.attention = attention;
        this.writeKv = writeKv;
        this.rope = rope;
      }

      private int lastKeyLength = -1;

      private LayerPlan plan(final TornadoExecutionPlan execution, final float[][] sources) {
        return new LayerPlan(
          execution, sources, this.x, this.residual, this.angles, this.perLayer,
          this.hiddenOut, this.residualOut, this.rotatedKey, this.storedValue,
          this.keyCache, this.valueCache, this.params, this.slots,
          this.hidden, this.keySize, this.cacheOffset, this.valueOffset, this.window,
          this.attention, this.writeKv, this.rope);
      }

      private void loadInput(final Layer layer, final int token) {
        int row = token * this.hidden;
        System.arraycopy(
          layer.hidden().data(), layer.hidden().offset() + row, this.x, 0, this.hidden);
        if (layer.residual() != null) {
          System.arraycopy(
            layer.residual().data(), layer.residual().offset() + row, this.residual, 0,
            this.hidden);
        }
      }

      private int prepare(final Layer layer, final int token) {
        if (this.perLayer != null) {
          System.arraycopy(
            layer.perLayerInput().data(),
            layer.perLayerInput().offset() + token * this.perLayer.length,
            this.perLayer, 0, this.perLayer.length);
        }
        if (!this.attention) {
          return 0;
        }
        int position = Math.round(layer.positions().get(token));
        this.rope.copyAngles(position, this.angles);
        int length = keyLength(layer.context(), token);
        int[] source = sourceSlots(layer.context(), length);
        int slot = layer.context().slotMapping()[token];
        if (length > this.slots.length || slot < 0) {
          throw new IllegalArgumentException("device cache cannot hold this token");
        }
        System.arraycopy(source, 0, this.slots, 0, length);
        this.params[POSITION] = position;
        this.params[SLOT] = slot;
        this.params[KEY_LENGTH] = length;
        this.params[HAS_RESIDUAL] = layer.shape() == HIGHWAY && layer.residual() != null ? 1 : 0;
        this.params[WINDOW] = this.window;
        this.params[CACHE_OFFSET] = this.cacheOffset;
        return length;
      }

      private boolean needsCacheRefresh(final int keyLength) {
        return this.writeKv && this.lastKeyLength >= 0 && keyLength <= this.lastKeyLength;
      }

      private void rememberKeyLength(final int keyLength) {
        if (this.writeKv) {
          this.lastKeyLength = keyLength;
        }
      }

      private void publish(final float[] hiddenDest, final float[] residualDest, final int row) {
        System.arraycopy(this.hiddenOut, 0, hiddenDest, row, this.hidden);
        if (residualDest != null) {
          System.arraycopy(this.residualOut, 0, residualDest, row, this.hidden);
        }
        if (!this.writeKv) {
          return;
        }
        int slot = this.params[SLOT];
        int keyBase = this.cacheOffset + slot * this.keySize;
        int valueBase = this.valueOffset + slot * this.keySize;
        System.arraycopy(this.rotatedKey, 0, this.keyCache, keyBase, this.keySize);
        System.arraycopy(this.storedValue, 0, this.valueCache, valueBase, this.keySize);
      }
    }

    boolean uses(final Set<float[]> weights) {
      for (float[] source : this.sources) {
        if (weights.contains(source)) {
          return true;
        }
      }
      return false;
    }

    void closeQuietly() {
      try {
        this.plan.close();
      } catch (TornadoExecutionPlanException failure) {
        requireNonNull(failure);
      }
    }

    static final class GroupPlan {
      private final TornadoExecutionPlan plan;
      private final List<Stage> stages;
      private final float[][] sources;
      private final int hidden;

      private GroupPlan(
        final TornadoExecutionPlan plan,
        final List<Stage> stages,
        final float[][] sources,
        final int hidden
      ) {
        this.plan = plan;
        this.stages = stages;
        this.sources = sources;
        this.hidden = hidden;
      }

      private void run(final List<Pending> batch) {
        int rows = batch.getFirst().layer.hidden().size(0);
        for (int token = 0; token < rows; token++) {
          this.stages.getFirst().loadInput(batch.getFirst().layer, token);
          int keyLength = 0;
          for (int index = 0; index < this.stages.size(); index++) {
            int length = this.stages.get(index).prepare(batch.get(index).layer, token);
            if (length > 0) {
              keyLength = length;
            }
          }
          TornadoLaunchLock.lock();
          try {
            for (Stage stage : this.stages) {
              if (stage.needsCacheRefresh(keyLength)) {
                this.plan.transferToDevice(stage.keyCache, stage.valueCache);
              }
            }
            this.plan.execute();
            for (Stage stage : this.stages) {
              stage.rememberKeyLength(keyLength);
            }
          } finally {
            TornadoLaunchLock.unlock();
          }
          int row = token * this.hidden;
          for (int index = 0; index < this.stages.size(); index++) {
            Tensor[] outputs = batch.get(index).outputs;
            this.stages.get(index).publish(
              outputs[0].data(), outputs.length > 1 ? outputs[1].data() : null, row);
          }
        }
      }

      private boolean uses(final Set<float[]> weights) {
        for (float[] source : this.sources) {
          if (weights.contains(source)) {
            return true;
          }
        }
        return false;
      }

      private void closeQuietly() {
        try {
          this.plan.close();
        } catch (TornadoExecutionPlanException failure) {
          requireNonNull(failure);
        }
      }
    }

    private static final class AttentionBuffers {
      private float[] angles;
      private int[] params;
      private int[] slots;
      private float[] rotatedKey;
      private float[] storedValue;
      private float[] keyCache;
      private float[] valueCache;
      private int keySize;
      private int cacheOffset;
      private int valueOffset;
      private float[] carried;
      private float[] mlpInput;
    }
  }

  private static final class Draft {
    private final GridScheduler scheduler = new GridScheduler();
    private final String name;
    private final Set<float[]> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    private TaskGraph graph;
    private String prefix = "";
    private float[] carried;
    private float[] perLayerRow;

    private void setPrefix(final String prefix) {
      this.prefix = prefix;
    }

    private String id(final String task) {
      return this.prefix + task;
    }

    private Draft(final String name) {
      this.name = name;
      this.graph = new TaskGraph(name);
    }

    private void resident(final float[] data) {
      if (data == null || !this.seen.add(data)) {
        return;
      }
      this.graph = this.graph.transferToDevice(FIRST_EXECUTION, data);
    }

    private void every(final float[] data) {
      if (data == null || !this.seen.add(data)) {
        return;
      }
      this.graph = this.graph.transferToDevice(EVERY_EXECUTION, data);
    }

    private void every(final int[] data) {
      this.graph = this.graph.transferToDevice(EVERY_EXECUTION, data);
    }

    private void host(final List<float[]> outputs) {
      this.graph =
        this.graph.transferToHost(EVERY_EXECUTION, (Object) outputs.toArray(float[][]::new));
    }

    private void gemv(
      final String task, final float[] input, final Matrix weight, final float[] output
    ) {
      if (input.length != weight.columns() || output.length != weight.rows()) {
        throw new IllegalArgumentException(task + " vector does not match the projection");
      }
      this.resident(weight.data());
      this.resident(output);
      float[] source = input;
      if (weight.inputScale() != 0f) {
        source = new float[input.length];
        float[] factor = {weight.inputScale()};
        this.resident(source);
        this.resident(factor);
        this.graph =
          this.graph.task(this.id(task + "In"), TornadoQwen3LayerKernels::srq, input, factor,
            source);
      }
      float[] bias = weight.bias() == null ? TornadoGemvKernels.NO_BIAS : weight.bias();
      this.resident(bias);
      float[] dest = output;
      if (weight.outputScale() != 0f) {
        dest = new float[output.length];
        this.resident(dest);
      }
      String name = this.id(task);
      WorkerGrid1D worker = new WorkerGrid1D(
        Math.multiplyExact(weight.rows(), TornadoGemvKernels.ROW_LANES));
      worker.setLocalWork(TornadoGemvKernels.ROW_LANES, 1, 1);
      this.scheduler.addWorkerGrid(this.name + "." + name, worker);
      this.graph = this.graph.task(
        name, TornadoGemvKernels::gemvTiled, new KernelContext(), source,
        weight.data(), weight.offset(), bias, weight.bias() == null ? 0 : 1, dest,
        weight.columns(), 0, weight.rows(), TornadoGemvKernels.ROW_LANES);
      if (dest != output) {
        float[] factor = {weight.outputScale()};
        this.resident(factor);
        this.graph = this.graph.task(
          this.id(task + "Out"), TornadoQwen3LayerKernels::srq, dest, factor, output);
      }
    }

    private TornadoExecutionPlan open() {
      return new TornadoExecutionPlan(this.graph.snapshot())
        .withGridScheduler(this.scheduler)
        .withoutPlanResultsHistory();
    }
  }

}
