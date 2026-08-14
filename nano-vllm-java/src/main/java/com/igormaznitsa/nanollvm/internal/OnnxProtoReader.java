package com.igormaznitsa.nanollvm.internal;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal ONNX protobuf decoder for Tier A: graph initializers and MatMul weight aliases.
 *
 * <p>Does not execute operators. Supports legacy {@code TensorProto.raw_data} field number
 * {@code 9} (pre-ONNX rename to {@code 13}), which transformers.js / onnx-community exports still
 * use. File size caps: 8 GiB on disk; in-memory decode requires ≤ ~2 GiB.
 *
 * @since 1.1.0
 */
public final class OnnxProtoReader {

  /**
   * ONNX TensorProto {@code FLOAT} (1).
   *
   * @since 1.1.0
   */
  public static final int FLOAT = OnnxDataTypes.FLOAT;
  /**
   * ONNX TensorProto {@code FLOAT16} (10).
   *
   * @since 1.1.0
   */
  public static final int FLOAT16 = OnnxDataTypes.FLOAT16;
  /**
   * ONNX TensorProto {@code DOUBLE} (11).
   *
   * @since 1.1.0
   */
  public static final int DOUBLE = OnnxDataTypes.DOUBLE;
  /**
   * ONNX TensorProto {@code BFLOAT16} (16).
   *
   * @since 1.1.0
   */
  public static final int BFLOAT16 = OnnxDataTypes.BFLOAT16;

  private static final long MAX_ONNX_BYTES = 8L * 1024 * 1024 * 1024;

  private OnnxProtoReader() {
  }

  /**
   * Reads the model graph from disk into initializers + MatMul aliases.
   *
   * @since 1.1.0
   */
  public static OnnxGraphBundle readGraph(final Path onnxFile) throws IOException {
    requireNonNull(onnxFile, "onnxFile");
    long size = Files.size(onnxFile);
    if (size > MAX_ONNX_BYTES) {
      throw new IOException("ONNX file exceeds max size (" + MAX_ONNX_BYTES + "): " + onnxFile);
    }
    if (size > Integer.MAX_VALUE) {
      throw new IOException("ONNX file larger than 2GiB buffer limit: " + onnxFile);
    }
    byte[] bytes = Files.readAllBytes(onnxFile);
    return readGraph(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN), onnxFile.toString());
  }

  /**
   * Convenience for {@link #readGraph(Path)}{@code .initializers()}.
   *
   * @since 1.1.0
   */
  public static List<OnnxTensorProto> readInitializers(final Path onnxFile) throws IOException {
    return readGraph(onnxFile).initializers();
  }

  /**
   * Convenience for {@link #readGraph(ByteBuffer, String)}{@code .initializers()}.
   *
   * @since 1.1.0
   */
  public static List<OnnxTensorProto> readInitializers(
    final ByteBuffer data,
    final String label
  ) throws IOException {
    return readGraph(data, label).initializers();
  }

  /**
   * Parses an in-memory ModelProto buffer ({@code label} is used in I/O error text).
   *
   * @since 1.1.0
   */
  public static OnnxGraphBundle readGraph(final ByteBuffer data, final String label)
    throws IOException {
    requireNonNull(data, "data");
    ByteBuffer buf = data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    buf.clear();
    List<OnnxTensorProto> initializers = new ArrayList<>();
    Map<String, String> matMulAliases = new LinkedHashMap<>();
    while (buf.hasRemaining()) {
      long tag = readVarint(buf, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (field == 7 && wire == 2) {
        parseGraph(readLengthDelimited(buf, label), label, initializers, matMulAliases);
      } else {
        skip(buf, wire, label);
      }
    }
    return new OnnxGraphBundle(List.copyOf(initializers), Map.copyOf(matMulAliases));
  }

  private static void parseGraph(
    final ByteBuffer graph,
    final String label,
    final List<OnnxTensorProto> initializers,
    final Map<String, String> matMulAliases
  ) throws IOException {
    while (graph.hasRemaining()) {
      long tag = readVarint(graph, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (field == 1 && wire == 2) {
        parseMatMulAlias(readLengthDelimited(graph, label), label, matMulAliases);
      } else if (field == 5 && wire == 2) {
        initializers.add(parseTensor(readLengthDelimited(graph, label), label));
      } else {
        skip(graph, wire, label);
      }
    }
  }

  private static void parseMatMulAlias(
    final ByteBuffer node,
    final String label,
    final Map<String, String> matMulAliases
  ) throws IOException {
    List<String> inputs = new ArrayList<>();
    String name = "";
    String opType = "";
    while (node.hasRemaining()) {
      long tag = readVarint(node, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (wire != 2) {
        skip(node, wire, label);
        continue;
      }
      String value = new String(readBytes(node, label), UTF_8);
      switch (field) {
        case 1 -> inputs.add(value);
        case 3 -> name = value;
        case 4 -> opType = value;
        default -> {
        }
      }
    }
    if (!"MatMul".equals(opType) || inputs.size() < 2 || name.isBlank()) {
      return;
    }
    String weightInit = inputs.get(1);
    if (weightInit.isBlank()) {
      return;
    }
    String hfName = matMulNodeToWeightName(name);
    if (hfName != null) {
      matMulAliases.putIfAbsent(weightInit, hfName);
    }
  }

  /**
   * {@code /model/layers.0/self_attn/q_proj/MatMul} → {@code model.layers.0.self_attn.q_proj.weight}
   *
   * @since 1.1.0
   */
  public static String matMulNodeToWeightName(final String nodeName) {
    String name = nodeName.strip();
    while (name.startsWith("/")) {
      name = name.substring(1);
    }
    String lower = name.toLowerCase(Locale.ROOT);
    if (lower.endsWith("/matmul")) {
      name = name.substring(0, name.length() - "/matmul".length());
    } else if (lower.endsWith("matmul")) {
      name = name.substring(0, name.length() - "matmul".length());
      while (name.endsWith("/") || name.endsWith(".")) {
        name = name.substring(0, name.length() - 1);
      }
    }
    if (name.isBlank()) {
      return null;
    }
    return name.replace('/', '.') + ".weight";
  }

  private static OnnxTensorProto parseTensor(final ByteBuffer tensor, final String label)
    throws IOException {
    List<Long> dims = new ArrayList<>();
    int dataType = 0;
    String name = "";
    byte[] rawData = null;
    float[] floatData = null;
    double[] doubleData = null;
    Map<String, String> external = new LinkedHashMap<>();
    while (tensor.hasRemaining()) {
      long tag = readVarint(tensor, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      switch (field) {
        case 1 -> {
          if (wire == 0) {
            dims.add(readVarint(tensor, label));
          } else if (wire == 2) {
            ByteBuffer packed = readLengthDelimited(tensor, label);
            while (packed.hasRemaining()) {
              dims.add(readVarint(packed, label));
            }
          } else {
            skip(tensor, wire, label);
          }
        }
        case 2 -> {
          if (wire != 0) {
            throw new IOException("TensorProto.data_type wire type " + wire + " in " + label);
          }
          dataType = (int) readVarint(tensor, label);
        }
        case 4 -> {
          if (wire == 2) {
            ByteBuffer packed = readLengthDelimited(tensor, label);
            int n = packed.remaining() / 4;
            floatData = new float[n];
            packed.asFloatBuffer().get(floatData);
          } else if (wire == 5) {
            floatData = appendFloat(floatData, Float.intBitsToFloat(tensor.getInt()));
          } else {
            skip(tensor, wire, label);
          }
        }
        case 8 -> {
          if (wire != 2) {
            throw new IOException("TensorProto.name wire type " + wire + " in " + label);
          }
          name = new String(readBytes(tensor, label), UTF_8);
        }
        case 9, 13 -> {
          // 9 = legacy raw_data; 13 = current onnx.proto raw_data
          if (wire != 2) {
            throw new IOException("TensorProto.raw_data wire type " + wire + " in " + label);
          }
          rawData = readBytes(tensor, label);
        }
        case 10 -> {
          if (wire == 2) {
            ByteBuffer packed = readLengthDelimited(tensor, label);
            int n = packed.remaining() / 8;
            doubleData = new double[n];
            packed.asDoubleBuffer().get(doubleData);
          } else if (wire == 1) {
            doubleData = appendDouble(doubleData, Double.longBitsToDouble(tensor.getLong()));
          } else {
            skip(tensor, wire, label);
          }
        }
        case 14 -> {
          if (wire != 2) {
            skip(tensor, wire, label);
          } else {
            parseExternalEntry(readLengthDelimited(tensor, label), label, external);
          }
        }
        default -> skip(tensor, wire, label);
      }
    }
    long[] dimArr = dims.stream().mapToLong(Long::longValue).toArray();
    return new OnnxTensorProto(name, dataType, dimArr, rawData, floatData, doubleData,
      Map.copyOf(external));
  }

  private static void parseExternalEntry(
    final ByteBuffer entry,
    final String label,
    final Map<String, String> external
  ) throws IOException {
    String key = null;
    String value = null;
    while (entry.hasRemaining()) {
      long tag = readVarint(entry, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (wire != 2) {
        skip(entry, wire, label);
        continue;
      }
      String s = new String(readBytes(entry, label), UTF_8);
      if (field == 1) {
        key = s;
      } else if (field == 2) {
        value = s;
      }
    }
    if (key != null && value != null) {
      external.put(key, value);
    }
  }

  private static float[] appendFloat(final float[] existing, final float value) {
    if (existing == null) {
      return new float[] {value};
    }
    float[] next = new float[existing.length + 1];
    System.arraycopy(existing, 0, next, 0, existing.length);
    next[existing.length] = value;
    return next;
  }

  private static double[] appendDouble(final double[] existing, final double value) {
    if (existing == null) {
      return new double[] {value};
    }
    double[] next = new double[existing.length + 1];
    System.arraycopy(existing, 0, next, 0, existing.length);
    next[existing.length] = value;
    return next;
  }

  private static ByteBuffer readLengthDelimited(final ByteBuffer buf, final String label)
    throws IOException {
    long len = readVarint(buf, label);
    if (len < 0 || len > buf.remaining()) {
      throw new IOException("invalid length-delimited field in " + label);
    }
    int start = buf.position();
    ByteBuffer slice = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    slice.position(start);
    slice.limit(start + (int) len);
    buf.position(start + (int) len);
    return slice.slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static byte[] readBytes(final ByteBuffer buf, final String label) throws IOException {
    long len = readVarint(buf, label);
    if (len < 0 || len > buf.remaining()) {
      throw new IOException("invalid bytes field in " + label);
    }
    byte[] out = new byte[(int) len];
    buf.get(out);
    return out;
  }

  private static void skip(final ByteBuffer buf, final int wire, final String label)
    throws IOException {
    switch (wire) {
      case 0 -> readVarint(buf, label);
      case 1 -> {
        if (buf.remaining() < 8) {
          throw new IOException("truncated 64-bit field in " + label);
        }
        buf.position(buf.position() + 8);
      }
      case 2 -> readLengthDelimited(buf, label);
      case 5 -> {
        if (buf.remaining() < 4) {
          throw new IOException("truncated 32-bit field in " + label);
        }
        buf.position(buf.position() + 4);
      }
      default -> throw new IOException("unsupported protobuf wire type " + wire + " in " + label);
    }
  }

  private static long readVarint(final ByteBuffer buf, final String label) throws IOException {
    long result = 0L;
    int shift = 0;
    while (true) {
      if (!buf.hasRemaining()) {
        throw new IOException("truncated varint in " + label);
      }
      int b = buf.get() & 0xFF;
      result |= (long) (b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return result;
      }
      shift += 7;
      if (shift > 63) {
        throw new IOException("varint too long in " + label);
      }
    }
  }

  /**
   * Parsed graph slice for Tier A import.
   *
   * @param initializers         named TensorProto weights / constants
   * @param matMulWeightAliases  anonymous MatMul {@code B} initializer name → HF-style
   *                             {@code ….weight} path derived from the MatMul node name
   * @since 1.1.0
   */
  public record OnnxGraphBundle(
    List<OnnxTensorProto> initializers,
    Map<String, String> matMulWeightAliases
  ) {
  }

  /**
   * Subset of ONNX {@code TensorProto} fields needed to decode float weights.
   *
   * @since 1.1.0
   */
  @SuppressWarnings("ArrayRecordComponent")
  public record OnnxTensorProto(
    String name,
    int dataType,
    long[] dims,
    byte[] rawData,
    float[] floatData,
    double[] doubleData,
    Map<String, String> externalData
  ) {
    /**
     * {@code true} when {@code external_data} includes a {@code location} key.
     *
     * @since 1.1.0
     */
    public boolean hasExternalData() {
      return this.externalData != null && this.externalData.containsKey("location");
    }
  }
}
