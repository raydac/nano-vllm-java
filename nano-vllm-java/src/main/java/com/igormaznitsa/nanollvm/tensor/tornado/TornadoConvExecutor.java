package com.igormaznitsa.nanollvm.tensor.tornado;

import static uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION;
import static uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * One compiled convolution plan per weight array and picture shape. Weights and bias stay on the
 * device. Each call copies the picture in and the filtered picture out, without packing windows.
 *
 * @since 1.5.0
 */
final class TornadoConvExecutor {

  private static final int MAX_CACHED_PLANS = 160;
  private static final Map<ConvKey, ConvPlan> CACHED_PLANS =
    new LinkedHashMap<>(MAX_CACHED_PLANS, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(final Map.Entry<ConvKey, ConvPlan> eldest) {
        if (this.size() <= MAX_CACHED_PLANS) {
          return false;
        }
        eldest.getValue().closeQuietly();
        return true;
      }
    };

  private TornadoConvExecutor() {
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
    if (drop.isEmpty()) {
      return;
    }
    TornadoLaunchLock.lock();
    try {
      CACHED_PLANS.entrySet().removeIf(entry -> {
        if (!drop.contains(entry.getKey().weight)) {
          return false;
        }
        entry.getValue().closeQuietly();
        return true;
      });
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  static boolean conv2d(
    final float[] input, final int inputOffset,
    final float[] weight, final int weightOffset,
    final float[] bias, final int biasOffset,
    final float[] output, final int outputOffset,
    final int inChannels, final int height, final int width,
    final int outChannels, final int kernel, final int stride, final int padding, final int groups,
    final boolean fuseSilu
  ) {
    if (!supported(
      inChannels, height, width, outChannels, kernel, stride, padding, groups)) {
      return false;
    }
    int outHeight = (height + 2 * padding - kernel) / stride + 1;
    int outWidth = (width + 2 * padding - kernel) / stride + 1;
    Kind kind = kind(inChannels, outChannels, kernel, stride, padding, groups);
    ConvKey key = new ConvKey(
      weight, weightOffset, bias, biasOffset, kind, inChannels, height, width, outChannels,
      kernel, stride, padding, groups, fuseSilu);
    TornadoLaunchLock.lock();
    try {
      ConvPlan plan = cached(key, outHeight, outWidth);
      System.arraycopy(input, inputOffset, plan.input, 0, plan.input.length);
      plan.execute();
      System.arraycopy(plan.output, 0, output, outputOffset, plan.output.length);
      return true;
    } catch (TornadoExecutionPlanException failure) {
      throw new IllegalStateException("TornadoVM convolution failed", failure);
    } finally {
      TornadoLaunchLock.unlock();
    }
  }

  private static boolean supported(
    final int inChannels, final int height, final int width, final int outChannels,
    final int kernel, final int stride, final int padding, final int groups
  ) {
    if (inChannels < 1 || height < 1 || width < 1 || outChannels < 1 || kernel < 1
      || stride < 1 || padding < 0 || groups < 1) {
      return false;
    }
    if (inChannels % groups != 0 || outChannels % groups != 0) {
      return false;
    }
    return height + 2L * padding >= kernel && width + 2L * padding >= kernel;
  }

  private static Kind kind(
    final int inChannels, final int outChannels, final int kernel, final int stride,
    final int padding, final int groups
  ) {
    if (kernel == 1 && stride == 1 && padding == 0 && groups == 1) {
      return Kind.POINTWISE;
    }
    if (kernel == 3 && padding == 1 && groups == 1 && (stride == 1 || stride == 2)) {
      return Kind.DENSE3;
    }
    if (kernel == 3 && padding == 1 && (stride == 1 || stride == 2)
      && groups == inChannels && inChannels == outChannels) {
      return Kind.DEPTHWISE3;
    }
    return Kind.GENERAL;
  }

  private static ConvPlan cached(final ConvKey key, final int outHeight, final int outWidth)
    throws TornadoExecutionPlanException {
    ConvPlan plan = CACHED_PLANS.get(key);
    if (plan != null) {
      return plan;
    }
    ConvPlan compiled = ConvPlan.compile(key, outHeight, outWidth);
    CACHED_PLANS.put(key, compiled);
    return compiled;
  }

  private enum Kind {
    POINTWISE,
    DENSE3,
    DEPTHWISE3,
    GENERAL
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record ConvKey(float[] weight, int weightOffset, float[] bias, int biasOffset, Kind kind,
                         int inChannels, int height, int width, int outChannels, int kernel,
                         int stride, int padding, int groups, boolean silu) {
  }

  private record ConvPlan(TornadoExecutionPlan plan, float[] input, float[] output) {
    static ConvPlan compile(final ConvKey key, final int outHeight, final int outWidth)
      throws TornadoExecutionPlanException {
      int inputLength =
        Math.multiplyExact(key.inChannels, Math.multiplyExact(key.height, key.width));
      int outputLength = Math.multiplyExact(
        key.outChannels, Math.multiplyExact(outHeight, outWidth));
      float[] input = new float[inputLength];
      float[] output = new float[outputLength];
      float[] bias = key.bias == null ? new float[key.outChannels] : key.bias;
      int biasOffset = key.bias == null ? 0 : key.biasOffset;
      int silu = key.silu ? 1 : 0;
      int[] spec = key.kind == Kind.GENERAL ? geometry(key, biasOffset, silu) : null;
      TaskGraph graph = new TaskGraph("nanollvm-conv")
        .transferToDevice(FIRST_EXECUTION, key.weight, bias)
        .transferToDevice(EVERY_EXECUTION, (Object) input);
      if (spec != null) {
        graph = graph.transferToDevice(FIRST_EXECUTION, (Object) spec);
      }
      graph = attach(graph, key, bias, biasOffset, silu, spec, input, output);
      TornadoExecutionPlan execution = TornadoPlans.open(
        graph.transferToHost(EVERY_EXECUTION, (Object) output).snapshot());
      return new ConvPlan(execution, input, output);
    }

    private static TaskGraph attach(
      final TaskGraph graph, final ConvKey key, final float[] bias, final int biasOffset,
      final int silu, final int[] spec, final float[] input, final float[] output
    ) {
      return switch (key.kind) {
        case POINTWISE -> graph.task(
          "run", TornadoConvKernels::pointwise,
          input, key.weight, bias, output, key.weightOffset, biasOffset,
          key.inChannels, key.height, key.width, key.outChannels, silu);
        case DENSE3 -> graph.task(
          "run", TornadoConvKernels::dense3x3,
          input, key.weight, bias, output, key.weightOffset, biasOffset,
          key.inChannels, key.height, key.width, key.outChannels, key.stride, silu);
        case DEPTHWISE3 -> graph.task(
          "run", TornadoConvKernels::depthwise3x3,
          input, key.weight, bias, output, key.weightOffset, biasOffset,
          key.inChannels, key.height, key.width, key.stride, silu);
        case GENERAL -> graph.task(
          "run", TornadoConvKernels::general, input, key.weight, bias, output, spec);
      };
    }

    private static int[] geometry(final ConvKey key, final int biasOffset, final int silu) {
      int outHeight = (key.height + 2 * key.padding - key.kernel) / key.stride + 1;
      int outWidth = (key.width + 2 * key.padding - key.kernel) / key.stride + 1;
      return new int[] {
        key.weightOffset, biasOffset, key.inChannels, key.height, key.width, key.outChannels,
        key.kernel, key.stride, key.padding, key.groups, silu, outHeight, outWidth
      };
    }

    void execute() throws TornadoExecutionPlanException {
      this.plan.execute();
    }

    void closeQuietly() {
      try {
        this.plan.close();
      } catch (TornadoExecutionPlanException failure) {
        Objects.requireNonNull(failure);
      }
    }
  }
}
