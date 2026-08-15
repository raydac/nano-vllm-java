package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;

/**
 * File-name helper that does not treat a root path as a tensor name.
 *
 * @since 1.1.0
 */
final class PathNames {

  private PathNames() {
  }

  /**
   * Basename of {@code path}, or the full string when the path has no file name.
   *
   * @since 1.1.0
   */
  static String of(final Path path) {
    Path name = requireNonNull(path, "path").getFileName();
    return name == null ? path.toString() : name.toString();
  }
}
