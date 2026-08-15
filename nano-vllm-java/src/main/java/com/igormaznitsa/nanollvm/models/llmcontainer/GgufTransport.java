package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * GGUF container transport: magic/header, metadata, tensor catalog, and payload bytes. Does not
 * interpret architecture families — that is {@link com.igormaznitsa.nanollvm.models.llmarch.ArchitectureProcessor}.
 *
 * @since 1.1.0
 */
public final class GgufTransport implements ContainerTransport, GgufTokenizerSource {

  private final GgufReader reader;
  private final String label;

  private GgufTransport(final GgufReader reader, final String label) {
    this.reader = requireNonNull(reader, "reader");
    this.label = requireNonNull(label, "label");
  }

  /**
   * Opens a {@code .gguf} file from disk.
   *
   * @param ggufPath path to the file
   * @return transport over that file
   * @since 1.1.0
   */
  public static GgufTransport open(final Path ggufPath) throws IOException {
    Path path = requireNonNull(ggufPath, "ggufPath").toAbsolutePath().normalize();
    return new GgufTransport(GgufReader.open(path), path.toString());
  }

  /**
   * Opens GGUF bytes already in memory ({@code virtualPath} is the display label).
   *
   * @since 1.1.0
   */
  public static GgufTransport open(final ByteBuffer data, final Path virtualPath)
    throws IOException {
    Path path = requireNonNull(virtualPath, "virtualPath").toAbsolutePath().normalize();
    return new GgufTransport(GgufReader.open(requireNonNull(data, "data"), path), path.toString());
  }

  public String label() {
    return this.label;
  }

  @Override
  public ContainerCatalog catalog() {
    return new ContainerCatalog(
      ModelSupport.Source.GGUF,
      this.label,
      this.reader.metaString("general.architecture", ""),
      this.reader.metadata(),
      ContainerCatalog.namesOf(this.reader.tensorNames()));
  }

  public Map<String, Object> readPayloads(
    final boolean allowUnpackParameters,
    final LlmListener io
  ) {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Map<String, Object> payloads = new LinkedHashMap<>();
    LoadProgress progress = new LoadProgress("GGUF weights", this.reader.tensorCount(), streams);
    long accountedBytes = 0L;
    try {
      for (String name : this.reader.tensorNames()) {
        if (allowUnpackParameters) {
          var tensor = this.reader.getTensor(name);
          payloads.put(name, tensor);
          accountedBytes += (long) tensor.numel() * Float.BYTES;
          progress.step(
            "%s (%.0f MiB float32)".formatted(name, accountedBytes / (1024.0 * 1024.0)));
        } else {
          PackedWeight weight = this.reader.getPackedWeight(name);
          payloads.put(name, weight);
          accountedBytes += weight.packedBytes();
          progress.step("%s (%.0f MiB packed)".formatted(name, accountedBytes / (1024.0 * 1024.0)));
        }
      }
      progress.finish(allowUnpackParameters
        ? "%.0f MiB float32 (unpacked at load)".formatted(accountedBytes / (1024.0 * 1024.0))
        : "%.0f MiB packed (dequant on matmul)".formatted(accountedBytes / (1024.0 * 1024.0)));
    } catch (RuntimeException e) {
      progress.finish("failed");
      throw e;
    }
    return payloads;
  }

  @Override
  public List<String> metaStringArray(final String key) {
    return this.reader.metaStringArray(key);
  }

  @Override
  public String metaString(final String key, final String defaultValue) {
    return this.reader.metaString(key, defaultValue);
  }

  @Override
  public int metaInt(final String key, final int defaultValue) {
    return this.reader.metaInt(key, defaultValue);
  }

  @Override
  public void close() throws IOException {
    this.reader.close();
  }
}
