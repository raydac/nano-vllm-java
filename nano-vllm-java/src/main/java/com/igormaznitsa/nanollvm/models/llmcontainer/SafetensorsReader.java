package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hugging Face {@code .safetensors} reader: JSON header plus float payloads. Files ≤ 2 GiB are
 * copied into a heap buffer so {@link #close()} can drop them; larger shards keep a positioned
 * {@link FileChannel}.
 *
 * @since 1.1.0
 */
public final class SafetensorsReader implements AutoCloseable {

  private static final ByteBuffer CLOSED_MAP = ByteBuffer.allocate(0).asReadOnlyBuffer();

  private final String label;
  private FileChannel channel;
  private ByteBuffer map;
  private final Map<String, TensorInfo> tensors;
  private final long dataOffset;
  private boolean closed;

  public SafetensorsReader(final Path path) throws IOException {
    requireNonNull(path, "path");
    this.label = path.toString();
    ResourceLimits limits = ResourceLimits.current();
    this.tensors = new LinkedHashMap<>();
    FileChannel opened = FileChannel.open(path);
    try {
      long size = opened.size();
      if (size <= Integer.MAX_VALUE) {
        this.map = ChannelBytes.readHeap(opened, size);
        opened.close();
        opened = null;
        this.channel = null;
        this.dataOffset = parseHeader(this.map, size, limits, this.label, this.tensors);
      } else {
        this.map = null;
        this.channel = opened;
        opened = null;
        this.dataOffset = parseHeader(this.channel, size, limits, this.label, this.tensors);
      }
    } finally {
      if (opened != null) {
        opened.close();
      }
    }
  }

  private SafetensorsReader(final String label, final ByteBuffer data) throws IOException {
    this.label = requireNonNull(label, "label");
    this.channel = null;
    ByteBuffer map = data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    map.clear();
    this.map = map;
    this.tensors = new LinkedHashMap<>();
    this.dataOffset = parseHeader(
      this.map, map.remaining(), ResourceLimits.current(), this.label, this.tensors);
  }

  private static long parseHeader(
    final ByteBuffer map,
    final long size,
    final ResourceLimits limits,
    final String label,
    final Map<String, TensorInfo> tensors
  ) throws IOException {
    if (size < 8) {
      throw new IOException("safetensors header truncated: " + label);
    }
    long headerLen = Integer.toUnsignedLong(map.getInt(0));
    requireHeaderFits(headerLen, size, limits, label);
    byte[] headerBytes = new byte[(int) headerLen];
    map.position(8);
    map.get(headerBytes);
    return fillTensors(headerBytes, tensors);
  }

  private static long parseHeader(
    final FileChannel channel,
    final long size,
    final ResourceLimits limits,
    final String label,
    final Map<String, TensorInfo> tensors
  ) throws IOException {
    if (size < 8) {
      throw new IOException("safetensors header truncated: " + label);
    }
    ByteBuffer prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    ChannelBytes.readFully(channel, 0, prefix);
    long headerLen = Integer.toUnsignedLong(prefix.getInt(0));
    requireHeaderFits(headerLen, size, limits, label);
    ByteBuffer header = ByteBuffer.allocate((int) headerLen);
    ChannelBytes.readFully(channel, 8, header);
    return fillTensors(header.array(), tensors);
  }

  private static void requireHeaderFits(
    final long headerLen,
    final long size,
    final ResourceLimits limits,
    final String label
  ) throws IOException {
    if (headerLen > limits.maxSafetensorsHeaderBytes()) {
      throw new IOException(
        "safetensors header length " + headerLen + " exceeds maxSafetensorsHeaderBytes ("
          + limits.maxSafetensorsHeaderBytes() + ")");
    }
    if (8L + headerLen > size) {
      throw new IOException("safetensors header extends past file end: " + label);
    }
  }

  private static long fillTensors(
    final byte[] headerBytes,
    final Map<String, TensorInfo> tensors
  ) throws IOException {
    long dataOffset = 8L + headerBytes.length;
    Map<String, Object> header = Json.parseObject(new String(headerBytes, UTF_8));
    for (var e : header.entrySet()) {
      if ("__metadata__".equals(e.getKey())) {
        continue;
      }
      Map<String, Object> info = Json.asObject(e.getValue());
      String dtype = Json.asString(info.get("dtype"));
      List<Object> shapeList = Json.asArray(info.get("shape"));
      List<Object> offsets = Json.asArray(info.get("data_offsets"));
      int[] shape = new int[shapeList.size()];
      for (int i = 0; i < shape.length; i++) {
        shape[i] = Json.asInt(shapeList.get(i), 0);
      }
      long start = Json.asLong(offsets.get(0), 0);
      long end = Json.asLong(offsets.get(1), 0);
      tensors.put(e.getKey(), new TensorInfo(dtype, shape, start, end));
    }
    return dataOffset;
  }

  public static SafetensorsReader open(final Path path) throws IOException {
    return new SafetensorsReader(path);
  }

  /**
   * Reads a safetensors blob from an in-memory buffer ({@code label} is used in error text).
   *
   * @since 1.1.0
   */
  public static SafetensorsReader open(final ByteBuffer data, final String label)
    throws IOException {
    return new SafetensorsReader(label, requireNonNull(data, "data"));
  }

  public static float float16ToFloat(final int h) {
    return Float.float16ToFloat((short) h);
  }

  public static float bfloat16ToFloat(final int h) {
    return Float.intBitsToFloat(h << 16);
  }

  public static List<Path> listSafetensors(final Path modelDir) throws IOException {
    try (var stream = Files.list(modelDir)) {
      return stream
        .filter(p -> PathNames.of(p).endsWith(".safetensors"))
        .sorted()
        .toList();
    }
  }

  public String label() {
    return this.label;
  }

  public Iterable<String> keys() {
    return this.tensors.keySet();
  }

  public int size() {
    return this.tensors.size();
  }

  public long byteSize(final String name) {
    TensorInfo info = this.requireInfo(name);
    return info.end - info.start;
  }

  public boolean contains(final String name) {
    return this.tensors.containsKey(name);
  }

  public Tensor getTensor(final String name) {
    this.requireOpen();
    TensorInfo info = this.requireInfo(name);
    int numel = 1;
    for (int d : info.shape) {
      numel = Math.multiplyExact(numel, d);
    }
    float[] data = new float[numel];
    ByteBuffer buf = this.slice(this.dataOffset + info.start, info.end - info.start);
    switch (info.dtype) {
      case "F32" -> buf.asFloatBuffer().get(data);
      case "F16" -> {
        short[] tmp = new short[numel];
        buf.asShortBuffer().get(tmp);
        for (int i = 0; i < numel; i++) {
          data[i] = Float.float16ToFloat(tmp[i]);
        }
      }
      case "BF16" -> {
        short[] tmp = new short[numel];
        buf.asShortBuffer().get(tmp);
        for (int i = 0; i < numel; i++) {
          data[i] = bfloat16ToFloat(tmp[i] & 0xFFFF);
        }
      }
      case "F64" -> {
        double[] tmp = new double[numel];
        buf.asDoubleBuffer().get(tmp);
        for (int i = 0; i < numel; i++) {
          data[i] = (float) tmp[i];
        }
      }
      default ->
        throw new UnsupportedOperationException("dtype " + info.dtype + " in " + this.label);
    }
    return Tensor.of(data, info.shape.length == 0 ? new int[] {1} : info.shape);
  }

  public String dtype(final String name) {
    return this.requireInfo(name).dtype;
  }

  public int[] shape(final String name) {
    return this.requireInfo(name).shape.clone();
  }

  public byte[] getRaw(final String name) {
    this.requireOpen();
    TensorInfo info = this.requireInfo(name);
    long length = info.end - info.start;
    if (length < 0 || length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("raw tensor too large: " + name);
    }
    byte[] data = new byte[(int) length];
    this.slice(this.dataOffset + info.start, length).get(data);
    return data;
  }

  private ByteBuffer slice(final long absoluteOffset, final long length) {
    if (length < 0 || length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("tensor slice too large in " + this.label);
    }
    int len = (int) length;
    if (this.map != null) {
      if (absoluteOffset > Integer.MAX_VALUE) {
        throw new IllegalArgumentException("tensor offset exceeds payload range in " + this.label);
      }
      int at = (int) absoluteOffset;
      ByteBuffer buf = this.map.duplicate().order(ByteOrder.LITTLE_ENDIAN);
      buf.position(at);
      buf.limit(at + len);
      return buf.slice().order(ByteOrder.LITTLE_ENDIAN);
    }
    ByteBuffer buf = ByteBuffer.allocate(len).order(ByteOrder.LITTLE_ENDIAN);
    try {
      ChannelBytes.readFully(this.channel, absoluteOffset, buf);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to read tensor slice from " + this.label, e);
    }
    buf.flip();
    return buf;
  }

  private TensorInfo requireInfo(final String name) {
    TensorInfo info = this.tensors.get(name);
    if (info == null) {
      throw new IllegalArgumentException("missing tensor: " + name);
    }
    return info;
  }

  @Override
  public void close() throws IOException {
    this.closed = true;
    this.map = CLOSED_MAP;
    FileChannel opened = this.channel;
    this.channel = null;
    if (opened != null) {
      opened.close();
    }
  }

  private void requireOpen() {
    if (this.closed) {
      throw new IllegalStateException("SafetensorsReader is closed: " + this.label);
    }
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record TensorInfo(String dtype, int[] shape, long start, long end) {
  }
}
