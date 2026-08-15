package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;

final class PathNames {

  private PathNames() {
  }

  static String of(final Path path) {
    Path name = requireNonNull(path, "path").getFileName();
    return name == null ? path.toString() : name.toString();
  }
}
