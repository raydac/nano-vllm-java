package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.LlmModelAccess;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Loaded causal LM: architecture weights, HF config, and tokenizer.
 *
 * <p>Safe to share across threads and across many {@link LLM} instances. Mutable inference
 * state (KV cache, scheduler, sampling) lives on each {@link LLM}, not here.
 *
 * <p>GGUF models keep quantized weights packed by default. Prefer unpacking at load with
 * {@link LlmModelFactory#make(Path, LlmListener, boolean)} ({@code true}) so float32 is built
 * directly from the mmap with no packed heap copy. {@link LLM.Builder#allowUnpackParameters()}
 * late-unpacks an already-packed model by installing a dense graph; packed tensors already bound
 * by existing engines are left intact (peak RAM may briefly hold packed + dense).
 *
 * <p>Construct via {@link LlmModelFactory#make(Path)}. Closing an {@link LLM} does not unload
 * this model — call {@link #close()} after every bound {@link LLM} is closed. The inference graph
 * is module-internal — applications use {@link LLM}, not a network accessor.
 *
 * @see LlmModelFactory
 * @see LLM
 */
public final class LlmModel implements AutoCloseable {

  static {
    LlmModelAccess.setResolver(LlmModel::resolveNetwork);
  }

  private final Path path;
  private final Config.HfConfig hfConfig;
  private final Tokenizer tokenizer;
  private final AtomicReference<WeightBag> weights;
  private final AtomicReference<CausalLM> network;
  private final ReentrantLock unpackLock = new ReentrantLock();
  private final AtomicBoolean closed = new AtomicBoolean();

  LlmModel(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final CausalLM network,
    final Tokenizer tokenizer
  ) {
    this.path = requireNonNull(path, "path").toAbsolutePath().normalize();
    this.hfConfig = requireNonNull(hfConfig, "hfConfig");
    this.weights = new AtomicReference<>(requireNonNull(weights, "weights"));
    this.network = new AtomicReference<>(requireNonNull(network, "network"));
    this.tokenizer = requireNonNull(tokenizer, "tokenizer");
  }

  /**
   * Filesystem path this model was loaded from (absolute, normalized).
   */
  public Path path() {
    this.assertNotClosed();
    return this.path;
  }

  /**
   * HuggingFace / GGUF-derived architecture config sealed at load.
   */
  public Config.HfConfig hfConfig() {
    this.assertNotClosed();
    return this.hfConfig;
  }

  /**
   * Tokenizer bound to this model; immutable and safe to share.
   */
  public Tokenizer tokenizer() {
    this.assertNotClosed();
    return this.tokenizer;
  }

  /**
   * Detected architecture label (e.g. {@code qwen3}, {@code gemma3}, {@code lfm2}).
   */
  public String architectureName() {
    return this.requireNetwork().architectureName();
  }

  /**
   * {@code true} when this model still holds GGUF-packed tensors that can be expanded to float32.
   */
  public boolean hasPackedWeights() {
    return this.requireWeights().hasPacked();
  }

  /**
   * {@code true} after {@link #close()} has released owned weight resources.
   */
  public boolean isClosed() {
    return this.closed.get();
  }

  /**
   * Releases owned weight resources (packed GGUF payloads and model-held graph refs).
   * Idempotent. Close every bound {@link LLM} first so engines drop their network references.
   */
  @Override
  public void close() {
    if (!this.closed.compareAndSet(false, true)) {
      return;
    }

    this.unpackLock.lock();
    try {
      WeightBag bag = this.weights.getAndSet(null);
      this.network.set(null);
      if (bag != null) {
        bag.releaseResources();
      }
    } finally {
      this.unpackLock.unlock();
    }
  }

  private CausalLM resolveNetwork(final boolean allowUnpackParameters, final LlmListener io) {
    CausalLM current = this.requireNetwork();
    if (!allowUnpackParameters) {
      return current;
    }
    if (!this.requireWeights().hasPacked()) {
      return current;
    }
    return this.unpackToDense(io == null ? LlmListeners.silent() : io);
  }

  private CausalLM unpackToDense(final LlmListener io) {
    this.unpackLock.lock();
    try {
      WeightBag currentWeights = this.requireWeights();
      if (!currentWeights.hasPacked()) {
        return this.requireNetwork();
      }

      LlmListeners.info(io, null, "Unpacking GGUF parameters to float32…");
      long startedAtNanos = System.nanoTime();
      WeightBag denseWeights = currentWeights.asDense();
      CausalLM built = CausalLMFactory.create(this.hfConfig, denseWeights);
      this.weights.set(denseWeights);
      this.network.set(built);
      LlmListeners.infof(io, null, "Unpacked float32 graph ready (%s) in %.1fs%n",
        built.architectureName(), (System.nanoTime() - startedAtNanos) / 1e9);
      return built;
    } finally {
      this.unpackLock.unlock();
    }
  }

  private CausalLM requireNetwork() {
    this.assertNotClosed();
    CausalLM current = this.network.get();
    if (current == null) {
      throw new IllegalStateException("LlmModel is closed");
    }
    return current;
  }

  private WeightBag requireWeights() {
    this.assertNotClosed();
    WeightBag bag = this.weights.get();
    if (bag == null) {
      throw new IllegalStateException("LlmModel is closed");
    }
    return bag;
  }

  private void assertNotClosed() {
    if (this.closed.get()) {
      throw new IllegalStateException("LlmModel is closed");
    }
  }
}
