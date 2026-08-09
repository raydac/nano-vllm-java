package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Skips resource-dependent tests when weights or corpus are absent, printing a visible warning
 * to stderr.
 */
public final class BundledModelAssumptions {

  private BundledModelAssumptions() {
  }

  public static Path requireQwen3() {
    return require(
      BundledModels.find(BundledModels.QWEN3_0_6B),
      "Qwen3-0.6B",
      "models/download-qwen3-0.6b.sh");
  }

  public static Path requireGemma3() {
    return require(
      BundledModels.find(BundledModels.GEMMA3_270M),
      "Gemma3-270M",
      "models/download-gemma3-270m.sh (HF license + HF_TOKEN)");
  }

  public static Path requireLfm2Gguf() {
    return require(
      BundledModels.find(BundledModels.LFM2_5_2_6B_GGUF),
      "LFM2.5-2.6B GGUF",
      "models/download-lfm2.5-2.6b-gguf.sh");
  }

  public static Path requireGteSmallGguf() {
    return require(
      BundledModels.find(BundledModels.GTE_SMALL_GGUF),
      "gte-small GGUF",
      "models/download-gte-small-gguf.sh");
  }

  public static Path requireBundledRag() {
    return require(
      BundledRag.find(),
      "bundled RAG corpus at " + BundledRag.ragRoot(),
      "create ./rag with corpus files or set -Dnanollvm.rag.dir=…");
  }

  public static Path require(
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
