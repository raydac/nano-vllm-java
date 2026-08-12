package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Path;
import java.util.Optional;

final class SampleModelAssumptions {

  private SampleModelAssumptions() {
  }

  static Path requireQwen3() {
    return require(
      BundledModels.find(BundledModels.QWEN3_0_6B),
      "Qwen3-0.6B",
      "models/download-qwen3-0.6b.sh");
  }

  static Path requireGteSmallGguf() {
    return require(
      BundledModels.find(BundledModels.GTE_SMALL_GGUF),
      "gte-small GGUF",
      "models/download-gte-small-gguf.sh");
  }

  static Path require(
    final Optional<Path> path,
    final String label,
    final String downloadHint
  ) {
    if (path.isPresent()) {
      return path.get();
    }
    String message = "Skipping test: %s not available (%s)".formatted(label, downloadHint);
    System.err.println("[nanollvm] WARNING: " + message);
    assumeTrue(false, message);
    throw new IllegalStateException("unreachable");
  }
}
