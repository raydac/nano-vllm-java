package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.models.LlmModel;

/**
 * In-module bridge from {@link com.igormaznitsa.nanollvm.engine.Transformer} / {@link
 * com.igormaznitsa.nanollvm.llm.LLM} to {@link LlmModel} without exporting {@link CausalLM} on the
 * application API.
 */
public final class LlmModelAccess {

  private static volatile Resolver resolver;
  private static volatile EngineLease engineLease;

  private LlmModelAccess() {
  }

  public static void setResolver(final Resolver resolver) {
    if (LlmModelAccess.resolver != null) {
      throw new IllegalStateException("LlmModelAccess resolver already set");
    }
    LlmModelAccess.resolver = resolver;
  }

  public static void setEngineLease(final EngineLease engineLease) {
    if (LlmModelAccess.engineLease != null) {
      throw new IllegalStateException("LlmModelAccess engine lease already set");
    }
    LlmModelAccess.engineLease = requireNonNull(engineLease, "engineLease");
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

  public static void acquireEngine(final LlmModel model) {
    LlmModelAccess.requireLease().acquire(model);
  }

  public static void releaseEngine(final LlmModel model) {
    LlmModelAccess.requireLease().release(model);
  }

  private static EngineLease requireLease() {
    EngineLease lease = LlmModelAccess.engineLease;
    if (lease == null) {
      throw new IllegalStateException("LlmModelAccess engine lease not initialized");
    }
    return lease;
  }

  @FunctionalInterface
  public interface Resolver {
    CausalLM resolveNetwork(
      LlmModel model,
      boolean allowUnpackParameters,
      LlmListener io
    );
  }

  public interface EngineLease {
    void acquire(LlmModel model);

    void release(LlmModel model);
  }
}
