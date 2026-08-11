package com.igormaznitsa.nanollvm.samples.utils;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_RAG_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_RAG_DIR;

import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
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
   * Resolution order: {@code -Dnanollvm.rag.dir} → {@code NANOLLVM_RAG_DIR} → {@code ./rag}.
   */
  public static Path ragRoot() {
    String prop = NanoLlvmProps.systemProperty(PROP_RAG_DIR);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_RAG_DIR);
    if (env != null) {
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
