package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class PiperOnnxNames {

  private static final Pattern RESBLOCK_CONVS = Pattern.compile(
    "^dec\\.resblocks\\.(\\d+)\\.convs\\.([01])\\.(weight|bias)$");

  private PiperOnnxNames() {
  }

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
