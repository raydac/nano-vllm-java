package com.igormaznitsa.nanollvm.models.llmcontainer;

import static com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes.BFLOAT16;
import static com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes.DOUBLE;
import static com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes.FLOAT;
import static com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes.FLOAT16;
import static com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsReader.bfloat16ToFloat;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;

/**
 * Decodes ONNX initializer payloads (embedded raw / typed arrays, or {@code external_data}) into
 * float32 {@link Tensor}s. Loadable types: FLOAT, FLOAT16, BFLOAT16, DOUBLE.
 *
 * @since 1.1.0
 */
public final class OnnxWeightReader {

  private OnnxWeightReader() {
  }

  /**
   * Materializes one floating initializer. {@code modelDir} is the base for external sidecars and
   * may be {@code null} when the payload is embedded.
   *
   * @since 1.1.0
   */
  public static Tensor toTensor(final OnnxTensorProto proto, final Path modelDir)
    throws IOException {
    requireNonNull(proto, "proto");
    if (!OnnxDataTypes.isLoadableFloatingWeight(proto.dataType())) {
      throw new UnsupportedOperationException(
        "unsupported ONNX data_type " + OnnxDataTypes.name(proto.dataType())
          + " (" + proto.dataType() + "); need FLOAT/FLOAT16/BFLOAT16/DOUBLE");
    }
    int[] shape = toIntShape(proto.dims());
    int numel = numel(shape);
    float[] data = decodePayload(proto, modelDir, numel);
    return Tensor.of(data, shape);
  }

  /**
   * Same as {@link OnnxDataTypes#isLoadableFloatingWeight(int)}.
   *
   * @since 1.1.0
   */
  public static boolean isFloatingWeightType(final int dataType) {
    return OnnxDataTypes.isLoadableFloatingWeight(dataType);
  }

  private static float[] decodePayload(
    final OnnxTensorProto proto,
    final Path modelDir,
    final int numel
  ) throws IOException {
    if (proto.hasExternalData()) {
      requireNonNull(modelDir, "modelDir");
      return decodeExternal(proto, modelDir, numel);
    }
    if (proto.rawData() != null) {
      return decodeRaw(proto.dataType(), proto.rawData(), numel);
    }
    if (proto.floatData() != null) {
      if (proto.floatData().length != numel) {
        throw new IOException(
          "float_data length " + proto.floatData().length + " != numel " + numel);
      }
      return proto.floatData().clone();
    }
    if (proto.doubleData() != null) {
      if (proto.doubleData().length != numel) {
        throw new IOException(
          "double_data length " + proto.doubleData().length + " != numel " + numel);
      }
      float[] out = new float[numel];
      for (int i = 0; i < numel; i++) {
        out[i] = (float) proto.doubleData()[i];
      }
      return out;
    }
    throw new IOException("tensor has no payload: " + proto.name());
  }

  private static float[] decodeExternal(
    final OnnxTensorProto proto,
    final Path modelDir,
    final int numel
  ) throws IOException {
    Map<String, String> ext = proto.externalData();
    String location = ext.get("location");
    if (location == null || location.isBlank()) {
      throw new IOException("external_data missing location for " + proto.name());
    }
    Path resolved = modelDir.resolve(location.replace('\\', '/')).normalize();
    Path base = modelDir.toAbsolutePath().normalize();
    if (!resolved.toAbsolutePath().normalize().startsWith(base)) {
      throw new IOException("external_data location escapes model dir: " + location);
    }
    if (!Files.isRegularFile(resolved)) {
      throw new IOException("missing external data file: " + resolved);
    }
    long offset = parseLong(ext.getOrDefault("offset", "0"), "offset");
    long length = parseLong(ext.get("length"), "length");
    int elementBytes = elementBytes(proto.dataType());
    if (length <= 0) {
      length = (long) numel * elementBytes;
    }
    if (length != (long) numel * elementBytes) {
      throw new IOException(
        "external length " + length + " != expected " + ((long) numel * elementBytes)
          + " for " + proto.name());
    }
    if (length > Integer.MAX_VALUE) {
      throw new IOException("external tensor larger than 2GiB buffer limit: " + proto.name());
    }
    byte[] raw = new byte[(int) length];
    try (FileChannel channel = FileChannel.open(resolved)) {
      if (offset + length > channel.size()) {
        throw new IOException("external data range past EOF: " + resolved);
      }
      ByteBuffer buf = ByteBuffer.wrap(raw);
      long read = 0L;
      while (read < length) {
        int n = channel.read(buf, offset + read);
        if (n < 0) {
          throw new IOException("unexpected EOF reading " + resolved);
        }
        read += n;
      }
    }
    return decodeRaw(proto.dataType(), raw, numel);
  }

  private static float[] decodeRaw(final int dataType, final byte[] raw, final int numel)
    throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
    float[] out = new float[numel];
    switch (dataType) {
      case FLOAT -> {
        if (raw.length < numel * 4) {
          throw new IOException("raw FLOAT truncated");
        }
        buf.asFloatBuffer().get(out);
      }
      case FLOAT16 -> {
        if (raw.length < numel * 2) {
          throw new IOException("raw FLOAT16 truncated");
        }
        for (int i = 0; i < numel; i++) {
          out[i] = Float.float16ToFloat(buf.getShort());
        }
      }
      case BFLOAT16 -> {
        if (raw.length < numel * 2) {
          throw new IOException("raw BFLOAT16 truncated");
        }
        for (int i = 0; i < numel; i++) {
          out[i] = bfloat16ToFloat(buf.getShort() & 0xFFFF);
        }
      }
      case DOUBLE -> {
        if (raw.length < numel * 8) {
          throw new IOException("raw DOUBLE truncated");
        }
        for (int i = 0; i < numel; i++) {
          out[i] = (float) buf.getDouble();
        }
      }
      default -> throw new UnsupportedOperationException(
        "unsupported ONNX data_type " + OnnxDataTypes.name(dataType)
          + " (" + dataType + "); need FLOAT/FLOAT16/BFLOAT16/DOUBLE");
    }
    return out;
  }

  private static int elementBytes(final int dataType) throws IOException {
    return switch (dataType) {
      case FLOAT -> 4;
      case FLOAT16, BFLOAT16 -> 2;
      case DOUBLE -> 8;
      default -> throw new IOException(
        "unsupported ONNX data_type for external data: " + OnnxDataTypes.name(dataType)
          + " (" + dataType + ")");
    };
  }

  private static int[] toIntShape(final long[] dims) throws IOException {
    int[] shape = new int[dims.length];
    for (int i = 0; i < dims.length; i++) {
      if (dims[i] < 0 || dims[i] > Integer.MAX_VALUE) {
        throw new IOException("invalid dim " + dims[i]);
      }
      shape[i] = (int) dims[i];
    }
    return shape;
  }

  private static int numel(final int[] shape) {
    int n = 1;
    for (int d : shape) {
      n = Math.multiplyExact(n, d);
    }
    return n;
  }

  private static long parseLong(final String text, final String field) throws IOException {
    if (text == null || text.isBlank()) {
      if ("length".equals(field)) {
        return -1L;
      }
      throw new IOException("external_data missing " + field);
    }
    try {
      return Long.parseLong(text.trim().toLowerCase(Locale.ROOT).replace("_", ""), 10);
    } catch (NumberFormatException e) {
      throw new IOException("invalid external_data " + field + ": " + text, e);
    }
  }
}
