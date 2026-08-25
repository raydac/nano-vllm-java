package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Piper ONNX initializer aliases: HiFi-GAN ResBlock2 {@code convs.0}/{@code convs.1} become
 * the PyTorch {@code convs1}/{@code convs2} names this graph expects. Official voices such as
 * Irina medium export the sequential residual pair that way.
 *
 * @since 1.3.0
 */
final class PiperOnnxNames {

  private static final Pattern RESBLOCK_CONVS = Pattern.compile(
    "^dec\\.resblocks\\.(\\d+)\\.convs\\.([01])\\.(weight|bias)$");

  private PiperOnnxNames() {
  }

  /**
   * Copies {@code convs.0}/{@code convs.1} residual weights onto {@code convs1}/{@code convs2}
   * names when those canonical keys are absent.
   */
  static void promoteDecoderResblocks(
    final Map<String, Tensor> tensors,
    final Map<String, ConvLayout> layouts
  ) {
    requireNonNull(tensors, "tensors");
    requireNonNull(layouts, "layouts");
    List.copyOf(tensors.keySet()).forEach(name -> {
      Matcher matcher = RESBLOCK_CONVS.matcher(name);
      if (!matcher.matches()) {
        return;
      }
      String canonical = "dec.resblocks.%s.convs%s.0.%s".formatted(
        matcher.group(1),
        "0".equals(matcher.group(2)) ? "1" : "2",
        matcher.group(3));
      tensors.putIfAbsent(canonical, tensors.get(name));
      ConvLayout layout = layouts.get(name);
      if (layout != null) {
        layouts.putIfAbsent(canonical, layout);
      }
    });
  }
}
