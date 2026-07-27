package io.nanovllm.tensor;

public abstract class FloatKernels {

  private static final FloatKernels INSTANCE = FloatKernelsFactory.create();

  protected FloatKernels() {
  }

  public static FloatKernels get() {
    return INSTANCE;
  }

  public abstract String name();

  public abstract float dot(float[] a, int aOffset, float[] b, int bOffset, int n);

  public abstract float sumSquares(float[] a, int offset, int n);

  public abstract void scaleAdd(
      float[] src, int srcOff, float[] weight, int wOff, float scale, float[] dst, int dstOff, int n
  );
}
