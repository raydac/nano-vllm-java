package com.igormaznitsa.nanollvm.exceptions;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * The checkpoint’s architecture or weight container is not one this library can run.
 *
 * @since 1.1.0
 */
public final class UnsupportedModelException extends ModelLoadException {

  private final String modelType;
  private final List<String> architectures;

  public UnsupportedModelException(
    final String message,
    final String modelType,
    final List<String> architectures
  ) {
    super(requireNonNull(message, "message"));
    this.modelType = modelType == null ? "" : modelType;
    this.architectures = architectures == null ? List.of() : List.copyOf(architectures);
  }

  public String modelType() {
    return this.modelType;
  }

  public List<String> architectures() {
    return this.architectures;
  }
}
