package com.igormaznitsa.nanollvm.weather.model;

/**
 * Raised before any weights are loaded when a required checkpoint is absent.
 */
public final class MissingModels extends IllegalStateException {

  MissingModels(final String message) {
    super(message);
  }
}
