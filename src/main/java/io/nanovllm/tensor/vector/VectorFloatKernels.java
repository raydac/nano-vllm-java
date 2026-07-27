package io.nanovllm.tensor.vector;

import io.nanovllm.tensor.FloatKernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorFloatKernels extends FloatKernels {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Override
  public String name() {
    return "Vector API %s (len=%d)".formatted(SPECIES, SPECIES.length());
  }

  @Override
  public float dot(float[] a, int aOffset, float[] b, int bOffset, int n) {
    int i = 0;
    FloatVector acc = FloatVector.zero(SPECIES);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + i);
      acc = va.fma(vb, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  @Override
  public float sumSquares(float[] a, int offset, int n) {
    int i = 0;
    FloatVector acc = FloatVector.zero(SPECIES);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector v = FloatVector.fromArray(SPECIES, a, offset + i);
      acc = v.fma(v, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  @Override
  public void scaleAdd(
      float[] src, int srcOff, float[] weight, int wOff, float scale, float[] dst, int dstOff, int n
  ) {
    int i = 0;
    FloatVector vScale = FloatVector.broadcast(SPECIES, scale);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector vs = FloatVector.fromArray(SPECIES, src, srcOff + i);
      FloatVector vw = FloatVector.fromArray(SPECIES, weight, wOff + i);
      vs.mul(vScale).mul(vw).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }
}
