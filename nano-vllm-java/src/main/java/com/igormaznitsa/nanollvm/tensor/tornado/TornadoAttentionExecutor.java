package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled attention plan per query length and key capacity. Lengths that change every token
 * are loaded from a buffer so the plan is not rebuilt for each new key.
 *
 * @since 1.5.0
 */
final class TornadoAttentionExecutor {

  private static final int MAX_CACHED_PLANS = 8;
  private static final int PARAMS = 9;
  private static final Map<PlanKey, AttendPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<PlanKey, AttendPlan> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private static volatile boolean unavailable;

  private TornadoAttentionExecutor() {
  }

  static boolean attend(
    final float[] query, final int queryOffset,
    final float[] key, final int keyOffset,
    final float[] value, final int valueOffset,
    final float[] result, final int resultOffset,
    final int queryStart, final int queryLength, final int keyIndexBase, final int keyLength,
    final int numHeads, final int numKvHeads, final int headDim,
    final float scale, final int slidingWindow,
    final boolean causal, final int[] keySlots
  ) {
    if (unavailable || queryLength <= 0 || keyLength < 0 || headDim <= 0
      || numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0) {
      return false;
    }
    if (keySlots != null && keySlots.length < keyLength) {
      return false;
    }
    int queryWidth = numHeads * headDim;
    int keyWidth = numKvHeads * headDim;
    int keyCapacity = ceilPowerOfTwo(Math.max(keyLength, 1));
    if (queryWidth == 0 || keyWidth == 0
      || keyCapacity > Integer.MAX_VALUE / keyWidth
      || queryLength > Integer.MAX_VALUE / queryWidth) {
      return false;
    }
    TornadoLaunchLock.lock();
    try {
      AttendPlan plan =
        cached(new PlanKey(numHeads, numKvHeads, headDim, queryLength, keyCapacity));
      plan.execute(
        query, queryOffset, key, keyOffset, value, valueOffset, result, resultOffset,
        queryStart, queryLength, keyIndexBase, keyLength, keyWidth,
        scale, slidingWindow, causal, keySlots
      );
      return true;
    } catch (Throwable failure) {
      if (failure instanceof OutOfMemoryError) {
        throw (OutOfMemoryError) failure;
      }
      unavailable = true;
      return false;
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static AttendPlan cached(final PlanKey key) throws TornadoExecutionPlanException {
    AttendPlan plan = CACHED_PLANS.get(key);
    if (plan == null) {
      plan = AttendPlan.compile(key);
      CACHED_PLANS.put(key, plan);
    }
    return plan;
  }

  private static int ceilPowerOfTwo(final int value) {
    int capacity = 1;
    while (capacity < value) {
      capacity <<= 1;
    }
    return capacity;
  }

  private record PlanKey(int numHeads, int numKvHeads, int headDim, int queryLength,
                         int keyCapacity) {
  }

  private static final class AttendPlan implements AutoCloseable {

    private final TornadoExecutionPlan plan;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] result;
    private final float[] params;
    private final int numHeads;
    private final int numKvHeads;
    private final int headDim;

    private AttendPlan(
      final TornadoExecutionPlan plan,
      final float[] query, final float[] key, final float[] value, final float[] result,
      final float[] params,
      final int numHeads, final int numKvHeads, final int headDim
    ) {
      this.plan = plan;
      this.query = query;
      this.key = key;
      this.value = value;
      this.result = result;
      this.params = params;
      this.numHeads = numHeads;
      this.numKvHeads = numKvHeads;
      this.headDim = headDim;
    }

    static AttendPlan compile(final PlanKey key) throws TornadoExecutionPlanException {
      int queryWidth = key.numHeads * key.headDim;
      int keyWidth = key.numKvHeads * key.headDim;
      float[] query = new float[key.queryLength * queryWidth];
      float[] keyScratch = new float[key.keyCapacity * keyWidth];
      float[] value = new float[key.keyCapacity * keyWidth];
      float[] result = new float[key.queryLength * queryWidth];
      float[] params = new float[PARAMS];
      params[0] = key.queryLength;
      params[2] = key.numHeads;
      params[3] = key.numKvHeads;
      params[4] = key.headDim;
      params[8] = Float.NEGATIVE_INFINITY;
      int work = key.queryLength * key.numHeads;
      int repeats = key.numHeads / key.numKvHeads;
      int[] headOfJob = new int[work];
      int[] queryOfJob = new int[work];
      int[] kvHeadOfJob = new int[work];
      for (int job = 0; job < work; job++) {
        int head = job / key.queryLength;
        headOfJob[job] = head;
        queryOfJob[job] = job - head * key.queryLength;
        kvHeadOfJob[job] = head / repeats;
      }
      TaskGraph graph = new TaskGraph("nanollvm-attend")
        .transferToDevice(FIRST_EXECUTION, headOfJob, queryOfJob, kvHeadOfJob)
        .transferToDevice(EVERY_EXECUTION, query, keyScratch, value, params)
        .task(
          "attend", TornadoAttentionKernels::scoreHeads,
          query, keyScratch, value, result, headOfJob, queryOfJob, kvHeadOfJob, params
        )
        .transferToHost(EVERY_EXECUTION, (Object) result);
      ImmutableTaskGraph snapshot = graph.snapshot();
      TornadoExecutionPlan execution = new TornadoExecutionPlan(snapshot).withPreCompilation();
      return new AttendPlan(execution, query, keyScratch, value, result, params,
        key.numHeads, key.numKvHeads, key.headDim);
    }

    void execute(
      final float[] query, final int queryOffset,
      final float[] key, final int keyOffset,
      final float[] value, final int valueOffset,
      final float[] result, final int resultOffset,
      final int queryStart, final int queryLength, final int keyIndexBase, final int keyLength,
      final int keyWidth,
      final float scale, final int slidingWindow, final boolean causal, final int[] keySlots
    ) throws TornadoExecutionPlanException {
      int queryWidth = this.numHeads * this.headDim;
      System.arraycopy(query, queryOffset + queryStart * queryWidth, this.query, 0,
        queryLength * queryWidth);
      this.packKeys(key, keyOffset, value, valueOffset, keyIndexBase, keyLength, keyWidth,
        keySlots);
      this.params[0] = queryLength;
      this.params[1] = keyLength;
      this.params[2] = this.numHeads;
      this.params[3] = this.numKvHeads;
      this.params[4] = this.headDim;
      this.params[5] = scale;
      this.params[6] = slidingWindow;
      this.params[7] = causal ? 1f : 0f;
      this.params[8] = Float.NEGATIVE_INFINITY;
      this.plan.execute();
      System.arraycopy(this.result, 0, result, resultOffset + queryStart * queryWidth,
        queryLength * queryWidth);
    }

    private void packKeys(
      final float[] key, final int keyOffset,
      final float[] value, final int valueOffset,
      final int keyIndexBase, final int keyLength, final int keyWidth, final int[] keySlots
    ) {
      if (keySlots == null) {
        System.arraycopy(key, keyOffset + keyIndexBase * keyWidth, this.key, 0,
          keyLength * keyWidth);
        System.arraycopy(value, valueOffset + keyIndexBase * keyWidth, this.value, 0,
          keyLength * keyWidth);
        return;
      }
      for (int token = 0; token < keyLength; token++) {
        int slot = keySlots[token];
        int destination = token * keyWidth;
        if (slot < 0) {
          Arrays.fill(this.key, destination, destination + keyWidth, 0f);
          Arrays.fill(this.value, destination, destination + keyWidth, 0f);
          continue;
        }
        System.arraycopy(key, keyOffset + slot * keyWidth, this.key, destination, keyWidth);
        System.arraycopy(value, valueOffset + slot * keyWidth, this.value, destination, keyWidth);
      }
    }

    @Override
    public void close() throws TornadoExecutionPlanException {
      this.plan.close();
    }

    void closeQuietly() {
      try {
        this.close();
      } catch (TornadoExecutionPlanException failure) {
        Objects.requireNonNull(failure);
      }
    }
  }
}
