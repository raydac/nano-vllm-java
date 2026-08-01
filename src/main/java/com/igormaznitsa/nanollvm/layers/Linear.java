package com.igormaznitsa.nanollvm.layers;

import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

public class Linear {

  protected Tensor weight;
  protected Tensor bias;

  public Linear(int inputSize, int outputSize, boolean bias) {
    this.weight = Tensor.zeros(outputSize, inputSize);
    this.bias = bias ? Tensor.zeros(outputSize) : null;
  }

  public Tensor weight() {
    return this.weight;
  }

  public void setWeight(Tensor weight) {
    this.weight = weight;
  }

  public Tensor bias() {
    return this.bias;
  }

  public void setBias(Tensor bias) {
    this.bias = bias;
  }

  public Tensor forward(Tensor x) {
    return Ops.linear(x, this.weight, this.bias);
  }

  public void loadWeight(Tensor loaded) {
    this.weight.copyFrom(loaded);
  }

  public static class Column extends Linear {
    public Column(int inputSize, int outputSize, boolean bias) {
      super(inputSize, outputSize, bias);
    }
  }

  public static final class Row extends Linear {
    public Row(int inputSize, int outputSize, boolean bias) {
      super(inputSize, outputSize, bias);
    }
  }

  public static final class Merged extends Column {
    private final int[] outputSizes;

    public Merged(int inputSize, int[] outputSizes, boolean bias) {
      super(inputSize, sum(outputSizes), bias);
      this.outputSizes = outputSizes.clone();
    }

    private static int sum(int[] a) {
      int s = 0;
      for (int v : a) {
        s += v;
      }
      return s;
    }

    public void loadShard(Tensor loaded, int shardId) {
      int shardOffset = 0;
      for (int i = 0; i < shardId; i++) {
        shardOffset += this.outputSizes[i];
      }
      int shardSize = this.outputSizes[shardId];
      int in = this.weight.size(1);
      for (int o = 0; o < shardSize; o++) {
        System.arraycopy(
            loaded.data(), loaded.offset() + o * in,
            this.weight.data(), (shardOffset + o) * in,
            in
        );
      }
    }
  }

  public static final class Qkv extends Column {
    private final int headSize;
    private final int numHeads;
    private final int numKvHeads;

    public Qkv(int hiddenSize, int headSize, int totalNumHeads, int totalNumKvHeads, boolean bias) {
      super(hiddenSize, (totalNumHeads + 2 * totalNumKvHeads) * headSize, bias);
      this.headSize = headSize;
      this.numHeads = totalNumHeads;
      this.numKvHeads = totalNumKvHeads;
    }

    public void loadShard(Tensor loaded, String shardId) {
      int shardSize;
      int shardOffset;
      switch (shardId) {
        case "q" -> {
          shardSize = this.numHeads * this.headSize;
          shardOffset = 0;
        }
        case "k" -> {
          shardSize = this.numKvHeads * this.headSize;
          shardOffset = this.numHeads * this.headSize;
        }
        case "v" -> {
          shardSize = this.numKvHeads * this.headSize;
          shardOffset = this.numHeads * this.headSize + this.numKvHeads * this.headSize;
        }
        default -> throw new IllegalArgumentException("shardId " + shardId);
      }
      int in = this.weight.size(1);
      for (int o = 0; o < shardSize; o++) {
        System.arraycopy(
            loaded.data(), loaded.offset() + o * in,
            this.weight.data(), (shardOffset + o) * in,
            in
        );
      }
    }
  }
}
