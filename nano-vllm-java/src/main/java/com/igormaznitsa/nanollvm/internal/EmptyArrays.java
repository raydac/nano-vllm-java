package com.igormaznitsa.nanollvm.internal;

/**
 * One shared zero-length array per element type. Nothing can be written into a zero-length
 * array, so every caller can hold the same instance.
 */
public final class EmptyArrays {

  public static final byte[] BYTES = new byte[0];
  public static final int[] INTS = new int[0];
  public static final float[] FLOATS = new float[0];
  public static final float[][] FLOAT_ROWS = new float[0][];

  private EmptyArrays() {
  }

  public static int[] ints(final int length) {
    return length == 0 ? INTS : new int[length];
  }

  public static float[] floats(final int length) {
    return length == 0 ? FLOATS : new float[length];
  }
}
