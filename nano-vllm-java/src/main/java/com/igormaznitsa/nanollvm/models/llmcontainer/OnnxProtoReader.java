package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.models.internal.ConvLayout.Kind;
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
 * Minimal ONNX protobuf decoder for Tier A: graph initializers and operator weight aliases.
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
   * Reads the model graph from disk into initializers + operator weight aliases.
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
    Map<String, String> identityAliases = new LinkedHashMap<>();
    Map<String, ConvLayout> convLayouts = new LinkedHashMap<>();
    while (buf.hasRemaining()) {
      long tag = readVarint(buf, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (field == 7 && wire == 2) {
        parseGraph(
          readLengthDelimited(buf, label),
          label,
          initializers,
          matMulAliases,
          identityAliases,
          convLayouts);
      } else {
        skip(buf, wire, label);
      }
    }
    return new OnnxGraphBundle(
      List.copyOf(initializers),
      Map.copyOf(matMulAliases),
      Map.copyOf(identityAliases),
      Map.copyOf(convLayouts));
  }

  private static void parseGraph(
    final ByteBuffer graph,
    final String label,
    final List<OnnxTensorProto> initializers,
    final Map<String, String> matMulAliases,
    final Map<String, String> identityAliases,
    final Map<String, ConvLayout> convLayouts
  ) throws IOException {
    while (graph.hasRemaining()) {
      long tag = readVarint(graph, label);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (field == 1 && wire == 2) {
        parseNodeAlias(
          readLengthDelimited(graph, label),
          label,
          matMulAliases,
          identityAliases,
          convLayouts);
      } else if (field == 5 && wire == 2) {
        initializers.add(parseTensor(readLengthDelimited(graph, label), label));
      } else {
        skip(graph, wire, label);
      }
    }
  }

  private static void parseNodeAlias(
    final ByteBuffer node,
    final String label,
    final Map<String, String> matMulAliases,
    final Map<String, String> identityAliases,
    final Map<String, ConvLayout> convLayouts
  ) throws IOException {
    List<String> inputs = new ArrayList<>();
    List<byte[]> attributes = new ArrayList<>();
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
      if (field == 5) {
        attributes.add(readBytes(node, label));
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
    if (name.isBlank()) {
      return;
    }
    if ("MatMul".equals(opType) && inputs.size() >= 2) {
      putAlias(matMulAliases, inputs.get(1), matMulNodeToWeightName(name));
      return;
    }
    if (isConvFamily(opType) && inputs.size() >= 2) {
      String weightName = convFamilyNodeToWeightName(name, opType);
      putAlias(identityAliases, inputs.get(1), weightName);
      putConvLayout(convLayouts, inputs.get(1), weightName, convLayout(opType, attributes, label));
      return;
    }
    if ("Gather".equals(opType) && !inputs.isEmpty() && isEmbeddingGather(name)) {
      putAlias(identityAliases, inputs.getFirst(), gatherNodeToWeightName(name));
    }
  }

  private static boolean isConvFamily(final String opType) {
    return "Conv".equals(opType) || "ConvTranspose".equals(opType);
  }

  private static String convFamilyNodeToWeightName(final String nodeName, final String opType) {
    return "ConvTranspose".equals(opType)
      ? convTransposeNodeToWeightName(nodeName)
      : convNodeToWeightName(nodeName);
  }

  private static void putConvLayout(
    final Map<String, ConvLayout> convLayouts,
    final String onnxName,
    final String canonical,
    final ConvLayout layout
  ) {
    if (onnxName != null && !onnxName.isBlank()) {
      convLayouts.putIfAbsent(onnxName, layout);
    }
    if (canonical != null && !canonical.isBlank()) {
      convLayouts.putIfAbsent(canonical, layout);
    }
  }

  private static ConvLayout convLayout(
    final String opType,
    final List<byte[]> attributes,
    final String label
  ) throws IOException {
    Map<String, int[]> ints = intAttributes(attributes, label);
    Kind kind = "ConvTranspose".equals(opType) ? Kind.CONV_TRANSPOSE : Kind.CONV;
    return new ConvLayout(
      kind,
      optionalPositive(ints, "strides"),
      optionalPadding(ints),
      optionalNonNegative(ints, "output_padding"),
      optionalPositive(ints, "dilations"),
      optionalPositive(ints, "group"));
  }

  private static Integer optionalPositive(final Map<String, int[]> ints, final String name) {
    int[] values = ints.get(name);
    if (values == null || values.length == 0 || values[0] < 1) {
      return null;
    }
    return values[0];
  }

  private static Integer optionalNonNegative(final Map<String, int[]> ints, final String name) {
    int[] values = ints.get(name);
    if (values == null || values.length == 0 || values[0] < 0) {
      return null;
    }
    return values[0];
  }

  private static Integer optionalPadding(final Map<String, int[]> ints) {
    int[] pads = ints.get("pads");
    if (pads == null || pads.length == 0) {
      return null;
    }
    int begin = pads.length <= 2 ? pads[0] : pads[pads.length / 2 - 1];
    return begin < 0 ? null : begin;
  }

  private static Map<String, int[]> intAttributes(
    final List<byte[]> attributes,
    final String label
  ) throws IOException {
    Map<String, int[]> out = new LinkedHashMap<>();
    for (byte[] payload : attributes) {
      ByteBuffer attr = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
      String attrName = "";
      List<Integer> values = new ArrayList<>();
      while (attr.hasRemaining()) {
        long tag = readVarint(attr, label);
        int field = (int) (tag >>> 3);
        int wire = (int) (tag & 7L);
        if (field == 1 && wire == 2) {
          attrName = new String(readBytes(attr, label), UTF_8);
        } else if (field == 3 && wire == 0) {
          values.add((int) readVarint(attr, label));
        } else if (field == 8 && wire == 0) {
          values.add((int) readVarint(attr, label));
        } else if ((field == 7 || field == 8) && wire == 2) {
          ByteBuffer packed = readLengthDelimited(attr, label);
          while (packed.hasRemaining()) {
            values.add((int) readVarint(packed, label));
          }
        } else {
          skip(attr, wire, label);
        }
      }
      if (!attrName.isBlank() && !values.isEmpty()) {
        out.putIfAbsent(attrName, values.stream().mapToInt(Integer::intValue).toArray());
      }
    }
    return out;
  }

  private static void putAlias(
    final Map<String, String> aliases,
    final String onnxName,
    final String canonical
  ) {
    if (onnxName == null || onnxName.isBlank() || canonical == null || canonical.equals(onnxName)) {
      return;
    }
    aliases.putIfAbsent(onnxName, canonical);
  }

  private static boolean isEmbeddingGather(final String nodeName) {
    return nodeName.toLowerCase(Locale.ROOT).contains("/emb/gather");
  }

  /**
   * {@code /model/layers.0/self_attn/q_proj/MatMul} → {@code model.layers.0.self_attn.q_proj.weight}
   *
   * @since 1.1.0
   */
  public static String matMulNodeToWeightName(final String nodeName) {
    return operatorNodeToWeightName(nodeName, "/matmul", "matmul");
  }

  /**
   * {@code /flow/flows.0/enc/in_layers.0/Conv} → {@code flow.flows.0.enc.in_layers.0.weight}
   *
   * @since 1.3.0
   */
  public static String convNodeToWeightName(final String nodeName) {
    return operatorNodeToWeightName(nodeName, "/conv", "conv");
  }

  /**
   * {@code /dec/ups.0/ConvTranspose} → {@code dec.ups.0.weight}
   *
   * @since 1.3.0
   */
  public static String convTransposeNodeToWeightName(final String nodeName) {
    return operatorNodeToWeightName(nodeName, "/convtranspose", "convtranspose");
  }

  /**
   * {@code /enc_p/emb/Gather} → {@code enc_p.emb.weight}
   *
   * @since 1.3.0
   */
  public static String gatherNodeToWeightName(final String nodeName) {
    return operatorNodeToWeightName(nodeName, "/gather", "gather");
  }

  private static String operatorNodeToWeightName(
    final String nodeName,
    final String slashSuffix,
    final String suffix
  ) {
    String name = nodeName.strip();
    while (name.startsWith("/")) {
      name = name.substring(1);
    }
    String lower = name.toLowerCase(Locale.ROOT);
    if (lower.endsWith(slashSuffix)) {
      name = name.substring(0, name.length() - slashSuffix.length());
    } else if (lower.endsWith(suffix)) {
      name = name.substring(0, name.length() - suffix.length());
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
   * @param initializers           named TensorProto weights / constants
   * @param matMulWeightAliases    anonymous MatMul {@code B} initializer name → HF-style
   *                               {@code ….weight} path derived from the MatMul node name
   * @param identityWeightAliases  Conv / ConvTranspose / embedding-Gather initializer name →
   *                               PyTorch-style {@code ….weight} path (no transpose)
   * @param convLayouts            Conv / ConvTranspose weight name → spatial attributes from the
   *                               consuming node (stride, pads, dilation, groups, output_padding)
   * @since 1.1.0
   */
  public record OnnxGraphBundle(
    List<OnnxTensorProto> initializers,
    Map<String, String> matMulWeightAliases,
    Map<String, String> identityWeightAliases,
    Map<String, ConvLayout> convLayouts
  ) {
    public OnnxGraphBundle {
      convLayouts = convLayouts == null ? Map.of() : Map.copyOf(convLayouts);
    }

    /**
     * Dilations greater than 1, keyed like {@link #convLayouts()}.
     *
     * @since 1.3.0
     */
    public Map<String, Integer> convDilations() {
      Map<String, Integer> out = new LinkedHashMap<>();
      this.convLayouts.forEach((name, layout) -> {
        int dilation = layout.dilationOr(1);
        if (dilation > 1) {
          out.put(name, dilation);
        }
      });
      return Map.copyOf(out);
    }
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
