package com.igormaznitsa.nanollvm.internal;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SafetensorsReader implements AutoCloseable {

  private final String label;
  private final FileChannel channel;
  private final ByteBuffer map;
  private final Map<String, TensorInfo> tensors;
  private final long dataOffset;

  public SafetensorsReader(final Path path) throws IOException {
    requireNonNull(path, "path");
    this.label = path.toString();
    ResourceLimits limits = ResourceLimits.current();
    this.channel = FileChannel.open(path);
    long size = this.channel.size();
    if (size > Integer.MAX_VALUE) {
      throw new IOException("safetensors larger than 2GiB mmap limit: " + path);
    }
    this.map =
      this.channel.map(FileChannel.MapMode.READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN);
    this.tensors = new LinkedHashMap<>();
    this.dataOffset = parseHeader(this.map, size, limits, this.label, this.tensors);
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
    if (headerLen > limits.maxSafetensorsHeaderBytes()) {
      throw new IOException(
        "safetensors header length " + headerLen + " exceeds maxSafetensorsHeaderBytes ("
          + limits.maxSafetensorsHeaderBytes() + ")");
    }
    if (8L + headerLen > size) {
      throw new IOException("safetensors header extends past file end: " + label);
    }
    byte[] headerBytes = new byte[(int) headerLen];
    map.position(8);
    map.get(headerBytes);
    long dataOffset = 8L + headerLen;
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
        .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
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
    TensorInfo info = this.requireInfo(name);
    int numel = 1;
    for (int d : info.shape) {
      numel = Math.multiplyExact(numel, d);
    }
    float[] data = new float[numel];
    long abs = this.dataOffset + info.start;
    ByteBuffer buf = this.map.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    buf.position((int) abs);
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
    return Tensor.of(data, info.shape);
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
    if (this.channel != null) {
      this.channel.close();
    }
  }

  private record TensorInfo(String dtype, int[] shape, long start, long end) {
  }
}
