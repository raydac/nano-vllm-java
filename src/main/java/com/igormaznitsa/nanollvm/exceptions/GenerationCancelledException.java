package com.igormaznitsa.nanollvm.exceptions;

public final class GenerationCancelledException extends NanoLlvmException {

  public GenerationCancelledException() {
    super("generation cancelled");
  }
}
