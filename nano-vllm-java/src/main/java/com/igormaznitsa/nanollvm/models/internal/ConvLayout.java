package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

/**
 * Spatial attributes of one ONNX {@code Conv} or {@code ConvTranspose} weight, taken from the
 * consuming node. Null numeric fields mean the attribute was omitted. Zero {@code padding}
 * is treated like omitted at use ({@link #paddingOr}) because ONNX often stores pad in a
 * sibling Pad node while this port fuses padding into the conv.
 *
 * @since 1.3.0
 */
public record ConvLayout(
  Kind kind,
  Integer stride,
  Integer padding,
  Integer outputPadding,
  Integer dilation,
  Integer groups
) {

  public ConvLayout {
    requireNonNull(kind, "kind");
    requirePositive(stride, "stride");
    requirePositive(dilation, "dilation");
    requirePositive(groups, "groups");
    requireNonNegative(padding, "padding");
    requireNonNegative(outputPadding, "outputPadding");
  }

  private static void requirePositive(final Integer value, final String name) {
    if (value != null && value < 1) {
      throw new IllegalArgumentException(name + " must be >= 1");
    }
  }

  private static void requireNonNegative(final Integer value, final String name) {
    if (value != null && value < 0) {
      throw new IllegalArgumentException(name + " must be >= 0");
    }
  }

  public boolean isTransposed() {
    return this.kind == Kind.CONV_TRANSPOSE;
  }

  public int strideOr(final int fallback) {
    return this.stride == null ? fallback : this.stride;
  }

  public int paddingOr(final int fallback) {
    return this.padding == null || this.padding == 0 ? fallback : this.padding;
  }

  public int outputPaddingOr(final int fallback) {
    return this.outputPadding == null ? fallback : this.outputPadding;
  }

  public int dilationOr(final int fallback) {
    return this.dilation == null ? fallback : this.dilation;
  }

  public int groupsOr(final int fallback) {
    return this.groups == null ? fallback : this.groups;
  }

  public enum Kind {
    CONV,
    CONV_TRANSPOSE
  }
}
