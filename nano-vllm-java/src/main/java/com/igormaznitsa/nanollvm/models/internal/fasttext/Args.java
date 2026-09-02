package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.io.IOException;

final class Args {

  static final int MODEL_CBOW = 1;
  static final int MODEL_SG = 2;
  static final int MODEL_SUP = 3;

  static final int LOSS_HS = 1;
  static final int LOSS_NS = 2;
  static final int LOSS_SOFTMAX = 3;
  static final int LOSS_OVA = 4;

  static final String DEFAULT_LABEL = "__label__";

  final int dim;
  final int ws;
  final int epoch;
  final int minCount;
  final int neg;
  final int wordNgrams;
  final int loss;
  final int model;
  final int bucket;
  final int minn;
  final int maxn;
  final int lrUpdateRate;
  final double t;
  final String label;
  boolean qout;

  private Args(
    final int dim,
    final int ws,
    final int epoch,
    final int minCount,
    final int neg,
    final int wordNgrams,
    final int loss,
    final int model,
    final int bucket,
    final int minn,
    final int maxn,
    final int lrUpdateRate,
    final double t
  ) {
    this.dim = dim;
    this.ws = ws;
    this.epoch = epoch;
    this.minCount = minCount;
    this.neg = neg;
    this.wordNgrams = wordNgrams;
    this.loss = loss;
    this.model = model;
    this.bucket = bucket;
    this.minn = minn;
    this.maxn = maxn;
    this.lrUpdateRate = lrUpdateRate;
    this.t = t;
    this.label = DEFAULT_LABEL;
    this.qout = false;
  }

  static Args load(final LittleEndianInput in, final int version) throws IOException {
    final int dim = in.readInt();
    final int ws = in.readInt();
    final int epoch = in.readInt();
    final int minCount = in.readInt();
    final int neg = in.readInt();
    final int wordNgrams = in.readInt();
    final int loss = in.readInt();
    final int model = in.readInt();
    final int bucket = in.readInt();
    final int minn = in.readInt();
    int maxn = in.readInt();
    final int lrUpdateRate = in.readInt();
    final double t = in.readDouble();
    if (version == 11 && model == MODEL_SUP) {
      maxn = 0;
    }
    return new Args(
      dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn, maxn, lrUpdateRate, t);
  }
}
