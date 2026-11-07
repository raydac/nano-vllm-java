package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.layers.Conv2d;
import com.igormaznitsa.nanollvm.layers.ConvTranspose2d;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Fused convolution: the ONNX file already folded batch-norm into the kernel.
 */
final class YoloConv {

  private final Conv2d conv;
  private final boolean silu;

  private YoloConv(final Conv2d conv, final boolean silu) {
    this.conv = conv;
    this.silu = silu;
  }

  static YoloConv load(
    final YoloTensors weights,
    final String prefix,
    final int inChannels,
    final int outChannels,
    final int kernel,
    final int stride,
    final int groups,
    final boolean silu
  ) {
    return new YoloConv(loadConv(
      weights, prefix + ".conv", inChannels, outChannels, kernel, stride, groups), silu);
  }

  static Conv2d loadBare(
    final YoloTensors weights,
    final String prefix,
    final int inChannels,
    final int outChannels,
    final int kernel,
    final int stride,
    final int groups
  ) {
    return loadConv(weights, prefix, inChannels, outChannels, kernel, stride, groups);
  }

  static ConvTranspose2d loadTranspose(
    final YoloTensors weights,
    final String prefix,
    final int channels,
    final int kernel,
    final int stride
  ) {
    return new ConvTranspose2d(
      weights.take(prefix + ".weight", channels, channels, kernel, kernel),
      weights.take(prefix + ".bias", channels),
      stride,
      0);
  }

  private static Conv2d loadConv(
    final YoloTensors weights,
    final String prefix,
    final int inChannels,
    final int outChannels,
    final int kernel,
    final int stride,
    final int groups
  ) {
    return new Conv2d(
      weights.take(prefix + ".weight", outChannels, inChannels / groups, kernel, kernel),
      weights.take(prefix + ".bias", outChannels),
      stride,
      kernel / 2,
      groups);
  }

  Tensor forward(final Tensor input, final MatmulRuntime runtime) {
    return this.conv.forward(input, runtime, this.silu);
  }
}
