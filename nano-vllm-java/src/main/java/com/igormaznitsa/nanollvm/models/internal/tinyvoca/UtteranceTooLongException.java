package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

/**
 * One tiny-voca call asked for more frames than the checkpoint allows.
 */
public final class UtteranceTooLongException extends IllegalArgumentException {

  public UtteranceTooLongException(final String message) {
    super(message);
  }
}
