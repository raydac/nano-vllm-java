package com.igormaznitsa.nanollvm.samples.utils;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.ENV_RAG_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_RAG_DIR;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Resolves the on-disk text RAG corpus directory (default {@code ./rag}).
 */
public final class BundledRag {

  public static final String DEFAULT_RAG_DIR = "rag";

  private BundledRag() {
  }

  /**
   * Resolution order: {@code -Dnanovllm.rag.dir} → {@code NANOVLLM_RAG_DIR} → {@code ./rag}.
   */
  public static Path ragRoot() {
    String prop = System.getProperty(PROP_RAG_DIR);
    if (prop != null && !prop.isBlank()) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = System.getenv(ENV_RAG_DIR);
    if (env != null && !env.isBlank()) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve(DEFAULT_RAG_DIR);
  }

  public static Optional<Path> find() {
    Path root = ragRoot();
    return Files.isDirectory(root) ? Optional.of(root) : Optional.empty();
  }

  public static Path require() {
    return find().orElseThrow(() -> new IllegalStateException(
        "RAG corpus not found at " + ragRoot()
            + " (create that folder or set -D" + PROP_RAG_DIR + "=… / " + ENV_RAG_DIR + ")"));
  }
}
