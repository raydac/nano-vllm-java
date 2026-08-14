package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * Orchestrates GGUF load: {@link GgufTransport} reads the container, {@link ModelBinding} selects
 * a supported graph (Qwen3 / LFM2 chat, BERT embeddings) and expected tensor names, then payloads
 * are copied. Default keeps large matrices GGML-packed; {@code allowUnpackParameters} dequantizes
 * each tensor to float32 from the mmap during load (no packed heap copy).
 */
public final class GgufModelLoader {

  private GgufModelLoader() {
  }

  public static LoadedGguf load(final Path ggufPath, final LlmListener io) throws IOException {
    return load(ggufPath, io, false);
  }

  public static LoadedGguf load(
    final Path ggufPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = requireNonNull(ggufPath, "ggufPath").toAbsolutePath().normalize();
    LlmListeners.infof(streams, null, "Loading GGUF from %s%n", path);
    return load(GgufTransport.open(path), streams, allowUnpackParameters);
  }

  /**
   * Loads GGUF weights from an in-memory buffer ({@code virtualPath} is the display label).
   *
   * @since 1.1.0
   */
  public static LoadedGguf load(
    final ByteBuffer data,
    final Path virtualPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = requireNonNull(virtualPath, "virtualPath").toAbsolutePath().normalize();
    LlmListeners.infof(streams, null, "Loading GGUF from memory (%s)%n", path);
    return load(GgufTransport.open(data, path), streams, allowUnpackParameters);
  }

  private static LoadedGguf load(
    final GgufTransport transport,
    final LlmListener streams,
    final boolean allowUnpackParameters
  ) throws IOException {
    try {
      ModelBinding.BoundModel bound = ModelBinding.bindGguf(transport.catalog());
      Config.HfConfig config = bound.config();
      LlmListeners.infof(streams, null,
        "GGUF %s: layers=%d hidden=%d ff=%d heads=%d/%d%n",
        bound.selection().architectureId(),
        config.numHiddenLayers(),
        config.hiddenSize(),
        config.intermediateSize(),
        config.numAttentionHeads(),
        config.numKeyValueHeads());
      WeightBag weights = transport.loadWeights(allowUnpackParameters, streams);
      bound.requireLoadedWeights(weights);
      return new LoadedGguf(config, weights, transport.reader(), bound.schema());
    } catch (UnsupportedModelException e) {
      closeTransport(transport);
      throw new UnsupportedModelException(
        "Cannot load GGUF '" + transport.label() + "'."
          + System.lineSeparator() + System.lineSeparator() + e.getMessage(),
        e.modelType(),
        e.architectures());
    } catch (RuntimeException e) {
      closeTransport(transport);
      throw e;
    }
  }

  private static void closeTransport(final GgufTransport transport) {
    try {
      transport.close();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  public record LoadedGguf(
    Config.HfConfig config,
    WeightBag weights,
    GgufReader reader,
    WeightSchema schema
  ) {
  }
}
