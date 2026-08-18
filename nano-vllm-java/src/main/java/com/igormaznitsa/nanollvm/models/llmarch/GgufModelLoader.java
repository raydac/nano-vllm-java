package com.igormaznitsa.nanollvm.models.llmarch;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufTransport;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * Orchestrates GGUF load: {@link GgufTransport} reads the container, {@link ModelBinding} selects
 * an {@link ArchitectureProcessor} (Qwen3 / LFM2 chat, BERT embeddings), then that processor fills
 * the weight bag. Default keeps large matrices GGML-packed;
 * {@code allowUnpackParameters} dequantizes each tensor to float32 from the file during load.
 *
 * @since 1.1.0
 */
public final class GgufModelLoader {

  private GgufModelLoader() {
  }

  /**
   * Loads a GGUF file with packed weights (no float32 unpack).
   *
   * @param ggufPath path to a {@code .gguf} file
   * @param io       load progress; {@code null} is treated as silent
   * @return config, weights, open transport, schema, and processor
   * @since 1.1.0
   */
  public static LoadedGguf load(final Path ggufPath, final LlmListener io) throws IOException {
    return load(ggufPath, io, false);
  }

  /**
   * Loads a GGUF file, optionally unpacking packed tensors to float32 during load.
   *
   * @param ggufPath              path to a {@code .gguf} file
   * @param io                    load progress; {@code null} is treated as silent
   * @param allowUnpackParameters {@code true} to dequantize to float32 at load
   * @return config, weights, open transport, schema, and processor
   * @since 1.1.0
   */
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
      ModelBinding.BoundModel bound = ModelBinding.bind(transport.catalog());
      Config.HfConfig config = bound.config();
      LlmListeners.infof(streams, null,
        "GGUF %s: layers=%d hidden=%d ff=%d heads=%d/%d%n",
        bound.selection().architectureId(),
        config.numHiddenLayers(),
        config.hiddenSize(),
        config.intermediateSize(),
        config.numAttentionHeads(),
        config.numKeyValueHeads());
      WeightBag weights = ModelFill.fill(transport, bound, streams, allowUnpackParameters);
      return new LoadedGguf(config, weights, transport, bound.schema(), bound.processor());
    } catch (UnsupportedModelException e) {
      closeTransport(transport);
      throw new UnsupportedModelException(
        "Cannot load GGUF '" + transport.label() + "'."
          + System.lineSeparator() + System.lineSeparator() + e.getMessage(),
        e.modelType(),
        e.architectures());
    } catch (RuntimeException | IOException e) {
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

  /**
   * Loaded GGUF: config, weights, live transport, schema, and architecture processor.
   *
   * @param config     GGUF metadata mapped onto Hugging Face-shaped config
   * @param weights    filled parameter bag
   * @param transport  open GGUF container (caller must close)
   * @param schema     expected parameter names
   * @param processor  family that bound and filled this file
   * @since 1.1.0
   */
  public record LoadedGguf(
    Config.HfConfig config,
    WeightBag weights,
    GgufTransport transport,
    WeightSchema schema,
    ArchitectureProcessor processor
  ) {

    public LoadedGguf {
      requireNonNull(config, "config");
      requireNonNull(weights, "weights");
      requireNonNull(transport, "transport");
      requireNonNull(schema, "schema");
      requireNonNull(processor, "processor");
    }
  }
}
