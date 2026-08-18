package com.igormaznitsa.nanollvm.models.llmcontainer;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.ModelFileBundle;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Hugging Face safetensors transport: {@code config.json} plus shard headers and payload bytes.
 * Does not rewrite names, merge Q/K/V, or pack Gemma 4 QAT — that is the architecture processor.
 *
 * @since 1.1.0
 */
public final class SafetensorsTransport implements ContainerTransport {

  private final String label;
  private final String configJson;
  private List<Shard> shards;
  private Map<String, Shard> shardsByLabel;
  private final List<TensorRef> tensorIndex;
  private final ContainerCatalog catalog;
  private boolean closed;

  private SafetensorsTransport(
    final String label,
    final String configJson,
    final List<Shard> shards,
    final List<TensorRef> tensorIndex
  ) {
    this.label = requireNonNull(label, "label");
    this.configJson = requireNonNull(configJson, "configJson");
    this.shards = List.copyOf(requireNonNull(shards, "shards"));
    Map<String, Shard> byLabel = new LinkedHashMap<>();
    this.shards.forEach(shard -> byLabel.put(shard.label(), shard));
    this.shardsByLabel = Map.copyOf(byLabel);
    this.tensorIndex = List.copyOf(requireNonNull(tensorIndex, "tensorIndex"));
    Set<String> names = new LinkedHashSet<>();
    this.tensorIndex.forEach(ref -> names.add(ref.name()));
    this.catalog = ContainerCatalog.ofHf(
      ModelSupport.Source.HF_SAFETENSORS, this.label, this.configJson, names);
  }

  /**
   * {@code true} when {@code modelDir} contains at least one {@code .safetensors} file.
   *
   * @since 1.1.0
   */
  public static boolean present(final Path modelDir) throws IOException {
    return !SafetensorsReader.listSafetensors(requireNonNull(modelDir, "modelDir")).isEmpty();
  }

  /**
   * Opens an HF folder: reads {@code config.json} and all {@code .safetensors} shards.
   *
   * @since 1.1.0
   */
  public static SafetensorsTransport open(final Path modelDir) throws IOException {
    Path dir = requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
    return open(dir, readConfigJson(dir));
  }

  /**
   * Opens an HF folder with a caller-supplied {@code config.json} body.
   *
   * @since 1.1.0
   */
  public static SafetensorsTransport open(final Path modelDir, final String configJson)
    throws IOException {
    Path dir = requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
    List<Path> files = SafetensorsReader.listSafetensors(dir);
    if (files.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors files in " + dir);
    }
    List<Shard> shards = files.stream()
      .map(file -> new Shard(PathNames.of(file), () -> SafetensorsReader.open(file)))
      .toList();
    return new SafetensorsTransport(dir.toString(), configJson, shards, probe(shards));
  }

  /**
   * Opens safetensors shards already in heap ({@code ModelFileSource} / classpath).
   *
   * @since 1.1.0
   */
  public static SafetensorsTransport open(
    final List<ModelFileBundle.NamedBytes> blobs,
    final String configJson,
    final String label
  ) throws IOException {
    requireNonNull(blobs, "blobs");
    if (blobs.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors blobs for " + label);
    }
    List<Shard> shards = blobs.stream()
      .map(blob -> new Shard(
        blob.name(),
        () -> SafetensorsReader.open(blob.buffer(), blob.name())))
      .toList();
    return new SafetensorsTransport(
      requireNonNull(label, "label"), configJson, shards, probe(shards));
  }

  private static String readConfigJson(final Path modelDir) throws IOException {
    Path configPath = modelDir.resolve(CONFIG_JSON);
    if (!Files.isRegularFile(configPath)) {
      throw new ModelLoadException("missing config.json in " + modelDir);
    }
    return Files.readString(configPath, UTF_8);
  }

  private static List<TensorRef> probe(final List<Shard> shards) throws IOException {
    List<TensorRef> index = new ArrayList<>();
    for (Shard shard : shards) {
      try (SafetensorsReader reader = shard.open()) {
        for (String name : reader.keys()) {
          index.add(new TensorRef(
            shard.label(),
            name,
            reader.dtype(name),
            reader.shape(name),
            reader.byteSize(name)));
        }
      }
    }
    return index;
  }

  public String label() {
    return this.label;
  }

  public String configJson() {
    return this.configJson;
  }

  public List<TensorRef> tensorIndex() {
    return this.tensorIndex;
  }

  public SafetensorsReader openShard(final String shardLabel) throws IOException {
    this.requireOpen();
    Shard shard = this.shardsByLabel.get(requireNonNull(shardLabel, "shardLabel"));
    if (shard == null) {
      throw new IllegalArgumentException("unknown safetensors shard: " + shardLabel);
    }
    return shard.open();
  }

  @Override
  public ContainerCatalog catalog() {
    return this.catalog;
  }

  @Override
  public void close() {
    this.closed = true;
    this.shards = List.of();
    this.shardsByLabel = Map.of();
  }

  private void requireOpen() {
    if (this.closed) {
      throw new IllegalStateException("SafetensorsTransport is closed: " + this.label);
    }
  }

  @FunctionalInterface
  interface ReaderOpen {
    SafetensorsReader open() throws IOException;
  }

  record Shard(String label, ReaderOpen opener) {
    SafetensorsReader open() throws IOException {
      return this.opener.open();
    }
  }

  /**
   * One tensor in a shard: name, dtype, shape, and payload size.
   *
   * @since 1.1.0
   */
  @SuppressWarnings("ArrayRecordComponent")
  public record TensorRef(
    String shardLabel,
    String name,
    String dtype,
    int[] shape,
    long byteSize
  ) {
    public TensorRef {
      requireNonNull(shardLabel, "shardLabel");
      requireNonNull(name, "name");
      requireNonNull(dtype, "dtype");
      requireNonNull(shape, "shape");
    }
  }
}
