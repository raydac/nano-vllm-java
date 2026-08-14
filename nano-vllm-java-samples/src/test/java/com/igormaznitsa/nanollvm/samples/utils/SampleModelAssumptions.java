package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Sample-module test helper: skip when optional local demo weights / corpus are absent.
 * Uses {@link BundledModels} / {@link BundledRag}; not part of the published library.
 */
public final class SampleModelAssumptions {

  private SampleModelAssumptions() {
  }

  public static Path requireQwen3() {
    return require(BundledModels.find(BundledModels.QWEN3_0_6B), "Qwen3-0.6B",
      "models/download-qwen3-0.6b.sh");
  }

  public static Path requireGemma3() {
    return require(BundledModels.find(BundledModels.GEMMA3_270M), "Gemma3-270M",
      "models/download-gemma3-270m.sh (HF license + HF_TOKEN)");
  }

  public static Path requireGemma4E2bQatMobile() {
    return require(BundledModels.find(BundledModels.GEMMA4_E2B_IT_QAT_MOBILE),
      "Gemma4-E2B-IT-QAT-Mobile",
      "models/download-gemma4-e2b-qat-mobile.sh");
  }

  public static Path requireGteSmallGguf() {
    return require(BundledModels.find(BundledModels.GTE_SMALL_GGUF), "gte-small GGUF",
      "models/download-gte-small-gguf.sh");
  }

  public static Path requireRag() {
    return require(BundledRag.find(), "local RAG corpus at " + BundledRag.ragRoot(),
      "create ./rag with corpus files or set -Dnanollvm.rag.dir=…");
  }

  public static Path require(final Optional<Path> path, final String label, final String hint) {
    if (path.isPresent()) {
      return path.get();
    }
    String message = "Skipping test: %s not available (%s)".formatted(label, hint);
    System.err.println("[nanollvm-samples] WARNING: " + message);
    assumeTrue(false, message);
    throw new IllegalStateException("unreachable");
  }
}
