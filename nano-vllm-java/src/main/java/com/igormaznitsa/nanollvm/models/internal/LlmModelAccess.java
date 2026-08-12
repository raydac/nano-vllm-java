package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.models.LlmModel;

/**
 * In-module bridge from {@link com.igormaznitsa.nanollvm.engine.Transformer} to {@link LlmModel}
 * without exporting {@link CausalLM} on the application API.
 */
public final class LlmModelAccess {

  private static volatile Resolver resolver;

  private LlmModelAccess() {
  }

  public static void setResolver(final Resolver resolver) {
    if (LlmModelAccess.resolver != null) {
      throw new IllegalStateException("LlmModelAccess resolver already set");
    }
    LlmModelAccess.resolver = resolver;
  }

  public static CausalLM resolveNetwork(
    final LlmModel model,
    final boolean allowUnpackParameters,
    final LlmListener io
  ) {
    Resolver active = LlmModelAccess.resolver;
    if (active == null) {
      throw new IllegalStateException("LlmModelAccess not initialized");
    }
    return active.resolveNetwork(model, allowUnpackParameters, io);
  }

  @FunctionalInterface
  public interface Resolver {
    CausalLM resolveNetwork(
      LlmModel model,
      boolean allowUnpackParameters,
      LlmListener io
    );
  }
}
