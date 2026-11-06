package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.layers.MaxPool2d;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * YOLO26 n/s blocks: bottleneck, C3k, C3k2, SPPF, and C2PSA.
 */
final class YoloBlocks {

  private YoloBlocks() {
  }

  private interface Block {
    Tensor forward(Tensor input, MatmulRuntime runtime);
  }

  static final class Bottleneck {
    private final YoloConv first;
    private final YoloConv second;
    private final boolean residual;

    private Bottleneck(final YoloConv first, final YoloConv second, final boolean residual) {
      this.first = first;
      this.second = second;
      this.residual = residual;
    }

    static Bottleneck load(
      final YoloTensors weights,
      final String prefix,
      final int channels,
      final int hidden,
      final int kernel,
      final boolean residual
    ) {
      return new Bottleneck(
        YoloConv.load(weights, prefix + ".cv1", channels, hidden, kernel, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv2", hidden, channels, kernel, 1, 1, true),
        residual);
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor out = this.second.forward(this.first.forward(input, runtime), runtime);
      return this.residual ? YoloMath.add(input, out) : out;
    }
  }

  static final class C3k {
    private final YoloConv left;
    private final YoloConv right;
    private final YoloConv fuse;
    private final Bottleneck[] blocks;

    private C3k(
      final YoloConv left,
      final YoloConv right,
      final YoloConv fuse,
      final Bottleneck[] blocks
    ) {
      this.left = left;
      this.right = right;
      this.fuse = fuse;
      this.blocks = blocks;
    }

    static C3k load(
      final YoloTensors weights,
      final String prefix,
      final int channels,
      final boolean residual
    ) {
      int hidden = channels / 2;
      Bottleneck[] blocks = new Bottleneck[2];
      for (int i = 0; i < blocks.length; i++) {
        blocks[i] = Bottleneck.load(
          weights, prefix + ".m." + i, hidden, hidden, 3, residual);
      }
      return new C3k(
        YoloConv.load(weights, prefix + ".cv1", channels, hidden, 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv2", channels, hidden, 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv3", hidden * 2, channels, 1, 1, 1, true),
        blocks);
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor branch = this.left.forward(input, runtime);
      for (Bottleneck block : this.blocks) {
        branch = block.forward(branch, runtime);
      }
      return this.fuse.forward(
        YoloMath.concatChannels(branch, this.right.forward(input, runtime)), runtime);
    }
  }

  static final class C3k2 {
    private final YoloConv expand;
    private final YoloConv project;
    private final Block[] blocks;
    private final int hidden;

    private C3k2(final YoloConv expand, final YoloConv project, final Block[] blocks,
                 final int hidden) {
      this.expand = expand;
      this.project = project;
      this.blocks = blocks;
      this.hidden = hidden;
    }

    static C3k2 load(
      final YoloTensors weights,
      final String prefix,
      final int inChannels,
      final int outChannels,
      final int repeats,
      final boolean c3k,
      final float expansion,
      final boolean attention
    ) {
      int hidden = (int) (outChannels * expansion);
      Block[] blocks = new Block[repeats];
      for (int i = 0; i < repeats; i++) {
        blocks[i] = block(weights, prefix + ".m." + i, hidden, c3k, attention);
      }
      return new C3k2(
        YoloConv.load(weights, prefix + ".cv1", inChannels, hidden * 2, 1, 1, 1, true),
        YoloConv.load(
          weights, prefix + ".cv2", hidden * (2 + repeats), outChannels, 1, 1, 1, true),
        blocks,
        hidden);
    }

    private static Block block(
      final YoloTensors weights,
      final String prefix,
      final int channels,
      final boolean c3k,
      final boolean attention
    ) {
      if (attention) {
        return new AttentionBlock(
          Bottleneck.load(weights, prefix + ".0", channels, channels / 2, 3, true),
          PsaBlock.load(weights, prefix + ".1", channels));
      }
      if (c3k) {
        return C3k.load(weights, prefix, channels, true)::forward;
      }
      return Bottleneck.load(weights, prefix, channels, channels / 2, 3, true)::forward;
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor expanded = this.expand.forward(input, runtime);
      Tensor[] parts = new Tensor[2 + this.blocks.length];
      parts[0] = YoloMath.sliceChannels(expanded, 0, this.hidden);
      parts[1] = YoloMath.sliceChannels(expanded, this.hidden, this.hidden);
      for (int i = 0; i < this.blocks.length; i++) {
        parts[i + 2] = this.blocks[i].forward(parts[i + 1], runtime);
      }
      return this.project.forward(YoloMath.concatChannels(parts), runtime);
    }
  }

  static final class Spff {
    private final YoloConv reduce;
    private final YoloConv project;
    private final MaxPool2d pool;
    private final int pools;
    private final boolean residual;

    private Spff(
      final YoloConv reduce,
      final YoloConv project,
      final MaxPool2d pool,
      final int pools,
      final boolean residual
    ) {
      this.reduce = reduce;
      this.project = project;
      this.pool = pool;
      this.pools = pools;
      this.residual = residual;
    }

    static Spff load(final YoloTensors weights, final String prefix, final int channels) {
      int hidden = channels / 2;
      int pools = 3;
      return new Spff(
        YoloConv.load(weights, prefix + ".cv1", channels, hidden, 1, 1, 1, false),
        YoloConv.load(weights, prefix + ".cv2", hidden * (pools + 1), channels, 1, 1, 1, true),
        new MaxPool2d(5, 1, 2),
        pools,
        true);
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor[] parts = new Tensor[this.pools + 1];
      parts[0] = this.reduce.forward(input, runtime);
      for (int i = 1; i < parts.length; i++) {
        parts[i] = this.pool.forward(parts[i - 1]);
      }
      Tensor fused = this.project.forward(YoloMath.concatChannels(parts), runtime);
      return this.residual ? YoloMath.add(input, fused) : fused;
    }
  }

  static final class C2Psa {
    private final YoloConv expand;
    private final YoloConv project;
    private final PsaBlock[] blocks;
    private final int hidden;

    private C2Psa(
      final YoloConv expand,
      final YoloConv project,
      final PsaBlock[] blocks,
      final int hidden
    ) {
      this.expand = expand;
      this.project = project;
      this.blocks = blocks;
      this.hidden = hidden;
    }

    static C2Psa load(
      final YoloTensors weights,
      final String prefix,
      final int channels,
      final int repeats
    ) {
      int hidden = channels / 2;
      PsaBlock[] blocks = new PsaBlock[repeats];
      for (int i = 0; i < repeats; i++) {
        blocks[i] = PsaBlock.load(weights, prefix + ".m." + i, hidden);
      }
      return new C2Psa(
        YoloConv.load(weights, prefix + ".cv1", channels, hidden * 2, 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv2", hidden * 2, channels, 1, 1, 1, true),
        blocks,
        hidden);
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor expanded = this.expand.forward(input, runtime);
      Tensor held = YoloMath.sliceChannels(expanded, 0, this.hidden);
      Tensor active = YoloMath.sliceChannels(expanded, this.hidden, this.hidden);
      for (PsaBlock block : this.blocks) {
        active = block.forward(active, runtime);
      }
      return this.project.forward(YoloMath.concatChannels(held, active), runtime);
    }
  }

  static final class PsaBlock {
    private final YoloAttention attention;
    private final YoloConv widen;
    private final YoloConv narrow;

    private PsaBlock(final YoloAttention attention, final YoloConv widen, final YoloConv narrow) {
      this.attention = attention;
      this.widen = widen;
      this.narrow = narrow;
    }

    static PsaBlock load(final YoloTensors weights, final String prefix, final int channels) {
      return new PsaBlock(
        YoloAttention.load(weights, prefix + ".attn", channels),
        YoloConv.load(weights, prefix + ".ffn.0", channels, channels * 2, 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".ffn.1", channels * 2, channels, 1, 1, 1, false));
    }

    Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      Tensor attended = YoloMath.add(input, this.attention.forward(input, runtime));
      Tensor fed = this.narrow.forward(this.widen.forward(attended, runtime), runtime);
      return YoloMath.add(attended, fed);
    }
  }

  private record AttentionBlock(Bottleneck bottleneck, PsaBlock attention) implements Block {

    @Override
    public Tensor forward(final Tensor input, final MatmulRuntime runtime) {
      return this.attention.forward(this.bottleneck.forward(input, runtime), runtime);
    }
  }
}
