package com.igormaznitsa.nanollvm.internal;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.ModelFileId;
import com.igormaznitsa.nanollvm.models.ModelFileSource;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Loads a {@link ModelFileSource} entirely into heap memory (no filesystem cache).
 *
 * @since 1.1.0
 */
public final class ModelFileBundle {

  private static final long MAX_WEIGHT_BYTES = 64L * 1024 * 1024 * 1024;
  private static final List<ModelFileId> HF_TEXT = List.of(
    ModelFileId.TOKENIZER,
    ModelFileId.TOKENIZER_CONFIG,
    ModelFileId.GENERATION_CONFIG,
    ModelFileId.ADDED_TOKENS,
    ModelFileId.SPECIAL_TOKENS_MAP);
  private static final List<ModelFileId> ONNX_IDS = List.of(
    ModelFileId.MODEL_ONNX,
    ModelFileId.MODEL_ONNX_FP16);

  private final String displayName;
  private final Path virtualPath;
  private final byte[] gguf;
  private final String configJson;
  private final Map<String, String> textFiles;
  private final byte[] sentencePieceModel;
  private final List<NamedBytes> safetensors;
  private final List<NamedBytes> onnx;

  private ModelFileBundle(
    final String displayName,
    final Path virtualPath,
    final byte[] gguf,
    final String configJson,
    final Map<String, String> textFiles,
    final byte[] sentencePieceModel,
    final List<NamedBytes> safetensors,
    final List<NamedBytes> onnx
  ) {
    this.displayName = displayName;
    this.virtualPath = virtualPath;
    this.gguf = gguf;
    this.configJson = configJson;
    this.textFiles = textFiles;
    this.sentencePieceModel = sentencePieceModel;
    this.safetensors = safetensors;
    this.onnx = onnx;
  }

  /**
   * Reads every listed {@link ModelFileId} from {@code source} into heap buffers.
   *
   * @since 1.1.0
   */
  public static ModelFileBundle load(final ModelFileSource source, final LlmListener io)
    throws IOException {
    requireNonNull(source, "source");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    LlmListeners.info(streams, null, "Loading model bytes from " + source.displayName() + "…");

    try (InputStream ggufIn = source.open(ModelFileId.GGUF)) {
      if (ggufIn != null) {
        byte[] bytes = readBounded(ggufIn, MAX_WEIGHT_BYTES);
        LlmListeners.infof(streams, null, "Read GGUF into memory (%.2f MiB)%n",
          bytes.length / (1024.0 * 1024.0));
        return new ModelFileBundle(
          source.displayName(),
          virtualPath(source.displayName()),
          bytes,
          null,
          Map.of(),
          null,
          List.of(),
          List.of());
      }
    }

    ResourceLimits limits = ResourceLimits.current();
    String configJson = new String(readRequired(source, ModelFileId.CONFIG, limits), UTF_8);
    Map<String, String> textFiles = new LinkedHashMap<>();
    for (ModelFileId id : HF_TEXT) {
      byte[] bytes = readOptional(source, id, limits);
      if (bytes != null) {
        textFiles.put(id.fileName(), new String(bytes, UTF_8));
      }
    }
    byte[] sentencePiece = readOptional(source, ModelFileId.TOKENIZER_MODEL, limits);

    List<NamedBytes> safetensors = readSafetensors(source, textFiles, limits);
    List<NamedBytes> onnx = List.of();
    if (safetensors.isEmpty()) {
      onnx = readOnnx(source);
      if (onnx.isEmpty()) {
        throw new ModelLoadException(
          "missing safetensors or ONNX weights from " + source.displayName());
      }
    }

    List<NamedBytes> weightFiles = safetensors.isEmpty() ? onnx : safetensors;
    long total = weightFiles.stream().mapToLong(w -> w.bytes().length).sum();
    LlmListeners.infof(streams, null, "Read HF model into memory (%.2f MiB, %d weight file%s)%n",
      total / (1024.0 * 1024.0),
      weightFiles.size(),
      weightFiles.size() == 1 ? "" : "s");

    return new ModelFileBundle(
      source.displayName(),
      virtualPath(source.displayName()),
      null,
      configJson,
      Map.copyOf(textFiles),
      sentencePiece,
      List.copyOf(safetensors),
      List.copyOf(onnx));
  }

  private static List<NamedBytes> readSafetensors(
    final ModelFileSource source,
    final Map<String, String> textFiles,
    final ResourceLimits limits
  ) throws IOException {
    List<NamedBytes> weights = new ArrayList<>();
    byte[] indexBytes = readOptional(source, ModelFileId.SAFE_TENSORS_INDEX, limits);
    if (indexBytes != null) {
      textFiles.put(ModelFileId.SAFE_TENSORS_INDEX.fileName(), new String(indexBytes, UTF_8));
      for (String shard : weightShards(indexBytes)) {
        try (InputStream in = source.openWeightShard(shard)) {
          if (in == null) {
            throw new ModelLoadException(
              "missing weight shard from " + source.displayName() + ": " + shard);
          }
          weights.add(new NamedBytes(shard, readBounded(in, MAX_WEIGHT_BYTES)));
        }
      }
      return weights;
    }
    try (InputStream in = source.open(ModelFileId.MODEL_SAFE_TENSORS)) {
      if (in != null) {
        weights.add(new NamedBytes(
          ModelFileId.MODEL_SAFE_TENSORS.fileName(),
          readBounded(in, MAX_WEIGHT_BYTES)));
      }
    }
    return weights;
  }

  private static List<NamedBytes> readOnnx(final ModelFileSource source) throws IOException {
    List<NamedBytes> weights = new ArrayList<>();
    for (ModelFileId id : ONNX_IDS) {
      try (InputStream in = source.open(id)) {
        if (in != null) {
          weights.add(new NamedBytes(id.fileName(), readBounded(in, MAX_WEIGHT_BYTES)));
          return weights;
        }
      }
    }
    return weights;
  }

  private static Path virtualPath(final String displayName) {
    String safe = displayName.replaceAll("[^a-zA-Z0-9._-]+", "_");
    if (safe.isBlank()) {
      safe = "stream";
    }
    return Path.of("/nanollvm-memory", safe).normalize();
  }

  private static Set<String> weightShards(final byte[] indexBytes) {
    Map<String, Object> root = Json.parseObject(new String(indexBytes, UTF_8));
    Map<String, Object> weightMap = Json.asObject(root.get("weight_map"));
    if (weightMap == null || weightMap.isEmpty()) {
      throw new ModelLoadException("SAFE_TENSORS_INDEX missing weight_map");
    }
    Set<String> shards = new LinkedHashSet<>();
    for (Object value : weightMap.values()) {
      String name = Json.asString(value);
      if (name == null || name.isBlank()) {
        throw new ModelLoadException("SAFE_TENSORS_INDEX weight_map has blank shard name");
      }
      if (name.indexOf('/') >= 0 || name.indexOf('\\') >= 0 || name.contains("..")) {
        throw new ModelLoadException("illegal weight shard name in index: " + name);
      }
      shards.add(name);
    }
    return shards;
  }

  private static byte[] readRequired(
    final ModelFileSource source,
    final ModelFileId id,
    final ResourceLimits limits
  ) throws IOException {
    try (InputStream in = source.open(id)) {
      if (in == null) {
        throw new ModelLoadException(
          "missing required model file " + id + " (" + id.fileName() + ") from "
            + source.displayName());
      }
      return readBounded(in, limits.maxFileBytes());
    }
  }

  private static byte[] readOptional(
    final ModelFileSource source,
    final ModelFileId id,
    final ResourceLimits limits
  ) throws IOException {
    try (InputStream in = source.open(id)) {
      return in == null ? null : readBounded(in, limits.maxFileBytes());
    }
  }

  private static byte[] readBounded(final InputStream in, final long maxBytes) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[64 * 1024];
    long copied = 0L;
    int n;
    while ((n = in.read(buffer)) >= 0) {
      copied += n;
      if (copied > maxBytes) {
        throw new ModelLoadException("model file exceeds max bytes (" + maxBytes + ")");
      }
      out.write(buffer, 0, n);
    }
    return out.toByteArray();
  }

  public boolean isGguf() {
    return this.gguf != null;
  }

  public String displayName() {
    return this.displayName;
  }

  public Path virtualPath() {
    return this.virtualPath;
  }

  public ByteBuffer ggufBuffer() {
    if (this.gguf == null) {
      throw new IllegalStateException("not a GGUF bundle");
    }
    return ByteBuffer.wrap(this.gguf);
  }

  public String configJson() {
    if (this.configJson == null) {
      throw new IllegalStateException("not an HF bundle");
    }
    return this.configJson;
  }

  public Optional<String> textFile(final ModelFileId id) {
    return Optional.ofNullable(this.textFiles.get(id.fileName()));
  }

  public Optional<String> textFile(final String fileName) {
    return Optional.ofNullable(this.textFiles.get(fileName));
  }

  /**
   * SentencePiece {@code tokenizer.model} bytes when present.
   *
   * @since 1.1.1
   */
  public Optional<byte[]> sentencePieceModel() {
    return this.sentencePieceModel == null
      ? Optional.empty()
      : Optional.of(this.sentencePieceModel.clone());
  }

  public List<NamedBytes> safetensors() {
    return this.safetensors;
  }

  /**
   * ONNX graph blobs from the source ({@code model.onnx} / {@code model_fp16.onnx} / {@code onnx/}).
   *
   * @since 1.1.0
   */
  public List<NamedBytes> onnx() {
    return this.onnx;
  }

  /**
   * Heap copy of a named weight or sidecar blob.
   *
   * @since 1.1.0
   */
  @SuppressWarnings("ArrayRecordComponent")
  public record NamedBytes(String name, byte[] bytes) {
    public NamedBytes {
      requireNonNull(name, "name");
      requireNonNull(bytes, "bytes");
    }

    public ByteBuffer buffer() {
      return ByteBuffer.wrap(this.bytes);
    }
  }
}
