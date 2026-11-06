package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.models.internal.yolo.YoloBlocks.C2Psa;
import com.igormaznitsa.nanollvm.models.internal.yolo.YoloBlocks.C3k2;
import com.igormaznitsa.nanollvm.models.internal.yolo.YoloBlocks.Spff;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Backbone and neck. Layer indexes match the YOLO26 YAML so weight names stay {@code model.N.*}.
 */
final class YoloTrunk {

  private final YoloConv stem;
  private final YoloConv p2;
  private final C3k2 b2;
  private final YoloConv p3Down;
  private final C3k2 b4;
  private final YoloConv p4Down;
  private final C3k2 b6;
  private final YoloConv p5Down;
  private final C3k2 b8;
  private final Spff spff;
  private final C2Psa psa;
  private final C3k2 h13;
  private final C3k2 h16;
  private final YoloConv h17;
  private final C3k2 h19;
  private final YoloConv h20;
  private final C3k2 h22;

  private YoloTrunk(
    final YoloConv stem,
    final YoloConv p2,
    final C3k2 b2,
    final YoloConv p3Down,
    final C3k2 b4,
    final YoloConv p4Down,
    final C3k2 b6,
    final YoloConv p5Down,
    final C3k2 b8,
    final Spff spff,
    final C2Psa psa,
    final C3k2 h13,
    final C3k2 h16,
    final YoloConv h17,
    final C3k2 h19,
    final YoloConv h20,
    final C3k2 h22
  ) {
    this.stem = stem;
    this.p2 = p2;
    this.b2 = b2;
    this.p3Down = p3Down;
    this.b4 = b4;
    this.p4Down = p4Down;
    this.b6 = b6;
    this.p5Down = p5Down;
    this.b8 = b8;
    this.spff = spff;
    this.psa = psa;
    this.h13 = h13;
    this.h16 = h16;
    this.h17 = h17;
    this.h19 = h19;
    this.h20 = h20;
    this.h22 = h22;
  }

  static YoloTrunk load(final YoloTensors weights, final YoloWidths width, final YoloTask task) {
    int repeats = width.repeat(2);
    boolean classify = task == YoloTask.CLASSIFY;
    boolean semantic = task == YoloTask.SEMANTIC;
    boolean neck = !classify;
    boolean pan = neck && !semantic;
    return new YoloTrunk(
      conv(weights, 0, 3, width.c64(), 2),
      conv(weights, 1, width.c64(), width.c128(), 2),
      c3(weights, 2, width.c128(), width.c256(), repeats, false, 0.25f, false),
      conv(weights, 3, width.c256(), width.c256(), 2),
      c3(weights, 4, width.c256(), width.c512(), repeats, false, 0.25f, false),
      conv(weights, 5, width.c512(), width.c512(), 2),
      c3(weights, 6, width.c512(), width.c512(), repeats, true, 0.5f, false),
      conv(weights, 7, width.c512(), width.c1024(), 2),
      c3(weights, 8, width.c1024(), width.c1024(), repeats, true, 0.5f, false),
      classify ? null : Spff.load(weights, "model.9", width.c1024()),
      C2Psa.load(weights, "model." + (classify ? 9 : 10), width.c1024(), repeats),
      neck ?
        c3(weights, 13, width.c1024() + width.c512(), width.c512(), repeats, true, 0.5f, false) :
        null,
      neck ?
        c3(weights, 16, width.c512() + width.c512(), width.c256(), repeats, true, 0.5f, false) :
        null,
      pan ? conv(weights, 17, width.c256(), width.c256(), 2) : null,
      pan ? c3(weights, 19, width.c256() + width.c512(), width.c512(), repeats, true, 0.5f, false) :
        null,
      pan ? conv(weights, 20, width.c512(), width.c512(), 2) : null,
      pan ? c3(weights, 22, width.c512() + width.c1024(), width.c1024(), 1, true, 0.5f, true) :
        null);
  }

  private static YoloConv conv(
    final YoloTensors weights,
    final int index,
    final int inChannels,
    final int outChannels,
    final int stride
  ) {
    return YoloConv.load(
      weights, "model." + index, inChannels, outChannels, 3, stride, 1, true);
  }

  private static C3k2 c3(
    final YoloTensors weights,
    final int index,
    final int inChannels,
    final int outChannels,
    final int repeats,
    final boolean c3k,
    final float expansion,
    final boolean attention
  ) {
    return C3k2.load(
      weights, "model." + index, inChannels, outChannels, repeats, c3k, expansion, attention);
  }

  List<Tensor> forward(final Tensor input, final MatmulRuntime runtime) {
    List<Tensor> layers = new ArrayList<>();
    layers.add(this.stem.forward(input, runtime));
    layers.add(this.p2.forward(layers.get(0), runtime));
    layers.add(this.b2.forward(layers.get(1), runtime));
    layers.add(this.p3Down.forward(layers.get(2), runtime));
    layers.add(this.b4.forward(layers.get(3), runtime));
    layers.add(this.p4Down.forward(layers.get(4), runtime));
    layers.add(this.b6.forward(layers.get(5), runtime));
    layers.add(this.p5Down.forward(layers.get(6), runtime));
    layers.add(this.b8.forward(layers.get(7), runtime));
    if (this.spff == null) {
      layers.add(this.psa.forward(layers.get(8), runtime));
      return layers;
    }
    layers.add(this.spff.forward(layers.get(8), runtime));
    layers.add(this.psa.forward(layers.get(9), runtime));
    if (this.h13 == null) {
      return layers;
    }
    layers.add(YoloMath.nearestTwice(layers.get(10)));
    layers.add(YoloMath.concatChannels(layers.get(11), layers.get(6)));
    layers.add(this.h13.forward(layers.get(12), runtime));
    layers.add(YoloMath.nearestTwice(layers.get(13)));
    layers.add(YoloMath.concatChannels(layers.get(14), layers.get(4)));
    layers.add(this.h16.forward(layers.get(15), runtime));
    if (this.h17 == null) {
      return layers;
    }
    layers.add(this.h17.forward(layers.get(16), runtime));
    layers.add(YoloMath.concatChannels(layers.get(17), layers.get(13)));
    layers.add(this.h19.forward(layers.get(18), runtime));
    layers.add(this.h20.forward(layers.get(19), runtime));
    layers.add(YoloMath.concatChannels(layers.get(20), layers.get(10)));
    layers.add(this.h22.forward(layers.get(21), runtime));
    return layers;
  }
}
