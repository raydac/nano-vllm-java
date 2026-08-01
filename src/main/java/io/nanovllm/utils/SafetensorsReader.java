package io.nanovllm.utils;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import io.nanovllm.tensor.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SafetensorsReader implements AutoCloseable {

  private final Path path;
  private final FileChannel channel;
  private final MappedByteBuffer map;
  private final Map<String, TensorInfo> tensors;
  private final long dataOffset;

  public SafetensorsReader(Path path) throws IOException {
    this.path = requireNonNull(path, "path");
    this.channel = FileChannel.open(path);
    long size = this.channel.size();
    this.map = this.channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
    this.map.order(ByteOrder.LITTLE_ENDIAN);
    long headerLen = Integer.toUnsignedLong(this.map.getInt(0));
    byte[] headerBytes = new byte[(int) headerLen];
    this.map.position(8);
    this.map.get(headerBytes);
    this.dataOffset = 8L + headerLen;
    Map<String, Object> header = Json.parseObject(new String(headerBytes, UTF_8));
    this.tensors = new LinkedHashMap<>();
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
      this.tensors.put(e.getKey(), new TensorInfo(dtype, shape, start, end));
    }
  }

  public static SafetensorsReader open(Path path) throws IOException {
    return new SafetensorsReader(path);
  }

  public static float float16ToFloat(int h) {
    return Float.float16ToFloat((short) h);
  }

  public static float bfloat16ToFloat(int h) {
    return Float.intBitsToFloat(h << 16);
  }

  public static List<Path> listSafetensors(Path modelDir) throws IOException {
    try (var stream = Files.list(modelDir)) {
      return stream
          .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
          .sorted()
          .toList();
    }
  }

  public Iterable<String> keys() {
    return this.tensors.keySet();
  }

  public int size() {
    return this.tensors.size();
  }

  public long byteSize(String name) {
    TensorInfo info = this.requireInfo(name);
    return info.end - info.start;
  }

  public boolean contains(String name) {
    return this.tensors.containsKey(name);
  }

  public Tensor getTensor(String name) {
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
      case "F32" -> {
        buf.asFloatBuffer().get(data);
      }
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
          throw new UnsupportedOperationException("dtype " + info.dtype + " in " + this.path);
    }
    return Tensor.of(data, info.shape);
  }

  private TensorInfo requireInfo(String name) {
    TensorInfo info = this.tensors.get(name);
    if (info == null) {
      throw new IllegalArgumentException("missing tensor: " + name);
    }
    return info;
  }

  @Override
  public void close() throws IOException {
    this.channel.close();
  }

  private record TensorInfo(String dtype, int[] shape, long start, long end) {
  }
}
