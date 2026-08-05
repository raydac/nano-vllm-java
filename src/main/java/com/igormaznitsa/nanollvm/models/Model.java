package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.nio.file.Path;

/**
 * Immutable loaded causal LM: architecture weights, HF config, and tokenizer.
 *
 * <p>Safe to share across threads and across many {@link LLM} instances. Mutable inference
 * state (KV cache, scheduler, sampling) lives on each {@link LLM}, not here.
 *
 * <p>Construct via {@link ModelFactory#make(Path)}. Closing an {@link LLM} does not unload
 * this model.
 *
 * @see ModelFactory
 * @see LLM
 */
public final class Model {

  private final Path path;
  private final Config.HfConfig hfConfig;
  private final CausalLM network;
  private final Tokenizer tokenizer;

  Model(Path path, Config.HfConfig hfConfig, CausalLM network, Tokenizer tokenizer) {
    this.path = requireNonNull(path, "path").toAbsolutePath().normalize();
    this.hfConfig = requireNonNull(hfConfig, "hfConfig");
    this.network = requireNonNull(network, "network");
    this.tokenizer = requireNonNull(tokenizer, "tokenizer");
  }

  public Path path() {
    return this.path;
  }

  public Config.HfConfig hfConfig() {
    return this.hfConfig;
  }

  public Tokenizer tokenizer() {
    return this.tokenizer;
  }

  public String architectureName() {
    return this.network.architectureName();
  }

  /**
   * Internal network graph used by the inference engine. Not part of the stable public surface
   * for application code.
   */
  public CausalLM network() {
    return this.network;
  }
}
