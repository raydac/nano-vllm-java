package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;

/**
 * GGUF v2/v3 reader: metadata KV map + named tensors (packed or float32). File payloads are
 * copied into a heap buffer so {@link #close()} can drop them without a native mmap.
 *
 * @since 1.1.0
 */
public final class GgufReader implements AutoCloseable, GgufTokenizerSource {

  private static final int GGUF_MAGIC = 0x46554747;
  private static final int DEFAULT_ALIGNMENT = 32;
  private static final int ARCHITECTURE_PEEK_BYTES = 4 * 1024 * 1024;
  private static final String META_ARCHITECTURE = "general.architecture";
  private static final ByteBuffer CLOSED_MAP = ByteBuffer.allocate(0).asReadOnlyBuffer();

  private final Path path;
  private final Map<String, Object> metadata;
  private final Map<String, TensorInfo> tensors;
  private final long tensorDataBase;
  private final ResourceLimits limits;
  private ByteBuffer map;
  private boolean closed;

  public GgufReader(final Path path) throws IOException {
    this.path = requireNonNull(path, "path").toAbsolutePath().normalize();
    this.limits = ResourceLimits.current();
    this.metadata = new LinkedHashMap<>();
    this.tensors = new LinkedHashMap<>();
    try (FileChannel channel = FileChannel.open(this.path)) {
      long size = channel.size();
      if (size > Integer.MAX_VALUE) {
        throw new IOException("GGUF larger than 2GiB heap read limit: " + this.path);
      }
      this.map = ChannelBytes.readHeap(channel, size);
    }
    this.tensorDataBase = this.parseContainer(this.map, this.map.capacity());
  }

  private GgufReader(final Path virtualPath, final ByteBuffer data) throws IOException {
    this.path = requireNonNull(virtualPath, "virtualPath").toAbsolutePath().normalize();
    this.limits = ResourceLimits.current();
    ByteBuffer map = data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    map.clear();
    this.map = map;
    this.metadata = new LinkedHashMap<>();
    this.tensors = new LinkedHashMap<>();
    this.tensorDataBase = this.parseContainer(this.map, map.remaining());
  }

  /**
   * Reads a GGUF container from an in-memory buffer ({@code virtualPath} is the display label).
   *
   * @since 1.1.0
   */
  public static GgufReader open(final ByteBuffer data, final Path virtualPath) throws IOException {
    return new GgufReader(virtualPath, requireNonNull(data, "data"));
  }

  public static GgufReader open(final Path path) throws IOException {
    return new GgufReader(path);
  }

  /**
   * Reads {@code general.architecture} from a metadata prefix (at most 4 MiB). Does not copy tensor
   * weights.
   *
   * @return architecture id, or empty when the key is absent from the prefix
   * @since 1.2.1
   */
  public static String peekArchitecture(final Path path) throws IOException {
    Path file = requireNonNull(path, "path").toAbsolutePath().normalize();
    try (FileChannel channel = FileChannel.open(file)) {
      long size = channel.size();
      if (size < 16) {
        throw new IOException("not a GGUF file (too small): " + file);
      }
      ByteBuffer prefix = ChannelBytes.readHeap(channel, Math.min(size, ARCHITECTURE_PEEK_BYTES));
      return readArchitecture(prefix, file);
    }
  }

  private static int[] toJavaShape(final List<Long> ggmlDims) {
    if (ggmlDims.isEmpty()) {
      return new int[] {1};
    }
    if (ggmlDims.size() == 1) {
      return new int[] {toIntDim(ggmlDims.getFirst())};
    }
    if (ggmlDims.size() == 2) {
      return new int[] {toIntDim(ggmlDims.get(1)), toIntDim(ggmlDims.get(0))};
    }
    int[] shape = new int[ggmlDims.size()];
    for (int i = 0; i < ggmlDims.size(); i++) {
      shape[i] = toIntDim(ggmlDims.get(ggmlDims.size() - 1 - i));
    }
    return shape;
  }

  private static int toIntDim(final long dim) {
    if (dim <= 0 || dim > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("invalid tensor dim " + dim);
    }
    return (int) dim;
  }

  private static long align(final long offset, final int alignment) {
    long rem = offset % alignment;
    return rem == 0 ? offset : offset + (alignment - rem);
  }

  private static String readArchitecture(final ByteBuffer map, final Path path) throws IOException {
    int magic = map.getInt(0);
    if (magic != GGUF_MAGIC) {
      throw new IOException("not a GGUF file (bad magic): " + path);
    }
    int version = map.getInt(4);
    if (version != 2 && version != 3) {
      throw new IOException("unsupported GGUF version " + version + " in " + path);
    }

    Cursor cursor = new Cursor(map, ResourceLimits.current(), 8);
    try {
      long tensorCount = cursor.readU64();
      long kvCount = cursor.readU64();
      if (tensorCount < 0 || tensorCount > 1_000_000 || kvCount < 0 || kvCount > 1_000_000) {
        throw new IOException("invalid GGUF counts in " + path);
      }
      for (long i = 0; i < kvCount; i++) {
        String key = cursor.readString();
        Object value = cursor.readValue(cursor.readU32AsInt());
        if (META_ARCHITECTURE.equals(key) && value instanceof String architecture) {
          return architecture;
        }
      }
      return "";
    } catch (IllegalStateException truncated) {
      throw new IOException(
        "GGUF metadata prefix too small to read general.architecture: " + path, truncated);
    }
  }

  private long parseContainer(final ByteBuffer map, final long size) throws IOException {
    int magic = map.getInt(0);
    if (magic != GGUF_MAGIC) {
      throw new IOException("not a GGUF file (bad magic): " + this.path);
    }
    int version = map.getInt(4);
    if (version != 2 && version != 3) {
      throw new IOException("unsupported GGUF version " + version + " in " + this.path);
    }

    Cursor cursor = new Cursor(map, this.limits, 8);
    long tensorCount = cursor.readU64();
    long kvCount = cursor.readU64();
    if (tensorCount < 0 || tensorCount > 1_000_000 || kvCount < 0 || kvCount > 1_000_000) {
      throw new IOException("invalid GGUF counts in " + this.path);
    }

    for (long i = 0; i < kvCount; i++) {
      String key = cursor.readString();
      int valueType = cursor.readU32AsInt();
      this.metadata.put(key, cursor.readValue(valueType));
    }

    int alignment = DEFAULT_ALIGNMENT;
    Object alignMeta = this.metadata.get("general.alignment");
    if (alignMeta instanceof Number n) {
      int candidate = n.intValue();
      if (candidate > 0) {
        alignment = candidate;
      }
    }

    for (long i = 0; i < tensorCount; i++) {
      String name = cursor.readString();
      int nDims = cursor.readU32AsInt();
      if (nDims < 0 || nDims > this.limits.maxGgufDims()) {
        throw new IOException(
          "GGUF tensor '" + name + "' nDims " + nDims + " exceeds maxGgufDims ("
            + this.limits.maxGgufDims() + ")");
      }
      long[] dimsRaw = new long[nDims];
      long numel = 1L;
      for (int d = 0; d < nDims; d++) {
        dimsRaw[d] = cursor.readU64();
        numel = Math.multiplyExact(numel, dimsRaw[d]);
      }
      int ggmlType = cursor.readU32AsInt();
      long relativeOffset = cursor.readU64();
      this.tensors.put(
        name,
        new TensorInfo(name, ggmlType, LongStream.of(dimsRaw).boxed().toList(), numel,
          relativeOffset));
    }

    long headerEnd = align(cursor.position, alignment);
    if (headerEnd > size) {
      throw new IOException("GGUF header exceeds file size: " + this.path);
    }
    return headerEnd;
  }

  public Path path() {
    return this.path;
  }

  public Map<String, Object> metadata() {
    return this.metadata;
  }

  public Object requireMeta(final String key) {
    Object value = this.metadata.get(key);
    if (value == null) {
      throw new IllegalArgumentException("missing GGUF metadata key: " + key);
    }
    return value;
  }

  @Override
  public String metaString(final String key, final String defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof String s ? s : defaultValue;
  }

  @Override
  public int metaInt(final String key, final int defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof Number n ? n.intValue() : defaultValue;
  }

  public long metaLong(final String key, final long defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof Number n ? n.longValue() : defaultValue;
  }

  public float metaFloat(final String key, final float defaultValue) {
    Object value = this.metadata.get(key);
    return value instanceof Number n ? n.floatValue() : defaultValue;
  }

  @Override
  public List<String> metaStringArray(final String key) {
    Object value = this.metadata.get(key);
    if (!(value instanceof List<?> list)) {
      return List.of();
    }
    return list.stream().map(Object::toString).toList();
  }

  public List<Number> metaNumberArray(final String key) {
    Object value = this.metadata.get(key);
    if (!(value instanceof List<?> list)) {
      return List.of();
    }
    List<Number> out = new ArrayList<>(list.size());
    for (Object item : list) {
      if (item instanceof Number n) {
        out.add(n);
      }
    }
    return out;
  }

  public Iterable<String> tensorNames() {
    return this.tensors.keySet();
  }

  public int tensorCount() {
    return this.tensors.size();
  }

  public boolean hasTensor(final String name) {
    return this.tensors.containsKey(name);
  }

  public TensorInfo info(final String name) {
    TensorInfo info = this.tensors.get(name);
    if (info == null) {
      throw new IllegalArgumentException("missing GGUF tensor: " + name);
    }
    return info;
  }

  /**
   * Loads and dequantizes a tensor to float32 from the file buffer (no owned packed {@code byte[]}
   * copy). 2D shapes are reversed to HF {@code [out, in]} / embedding {@code [vocab, dim]} layout.
   */
  public Tensor getTensor(final String name) {
    this.requireOpen();
    TensorInfo info = this.info(name);
    long abs = this.tensorDataBase + info.relativeOffset;
    if (abs > Integer.MAX_VALUE) {
      throw new IllegalStateException("tensor offset exceeds payload range: " + name);
    }
    long byteLen = GgufDequant.packedByteLength(info.ggmlType, info.numel);
    if (byteLen > Integer.MAX_VALUE) {
      throw new IllegalStateException("tensor bytes exceed int: " + name);
    }
    if (info.numel > Integer.MAX_VALUE) {
      throw new IllegalStateException("tensor numel exceeds int: " + name);
    }
    ByteBuffer payload = GgufDequant.littleEndianSlice(this.map, (int) abs, (int) byteLen);
    float[] data = GgufDequant.dequantize(payload, info.ggmlType, info.numel);
    return Tensor.of(data, toJavaShape(info.dims()));
  }

  /**
   * Loads a tensor keeping GGML blocks packed (owned byte copy). Shape uses HF layout for 2D.
   */
  public PackedWeight getPackedWeight(final String name) {
    this.requireOpen();
    TensorInfo info = this.info(name);
    long abs = this.tensorDataBase + info.relativeOffset;
    if (abs > Integer.MAX_VALUE) {
      throw new IllegalStateException("tensor offset exceeds payload range: " + name);
    }
    long byteLen = GgufDequant.packedByteLength(info.ggmlType, info.numel);
    if (byteLen > Integer.MAX_VALUE) {
      throw new IllegalStateException("tensor bytes exceed int: " + name);
    }
    ByteBuffer payload = GgufDequant.littleEndianSlice(this.map, (int) abs, (int) byteLen);
    byte[] packed = new byte[(int) byteLen];
    payload.get(packed);
    return new PackedWeight(packed, info.ggmlType, toJavaShape(info.dims()), info.numel);
  }

  @Override
  public void close() {
    this.closed = true;
    this.map = CLOSED_MAP;
  }

  private void requireOpen() {
    if (this.closed) {
      throw new IllegalStateException("GgufReader is closed: " + this.path);
    }
  }

  public record TensorInfo(
    String name,
    int ggmlType,
    List<Long> dims,
    long numel,
    long relativeOffset
  ) {
    public TensorInfo {
      dims = List.copyOf(dims);
    }
  }

  private static final class Cursor {
    private final ByteBuffer map;
    private final ResourceLimits limits;
    private long position;

    Cursor(final ByteBuffer map, final ResourceLimits limits, final long position) {
      this.map = map;
      this.limits = limits;
      this.position = position;
    }

    private void require(final int bytes) {
      if (this.position + bytes > this.map.capacity()) {
        throw new IllegalStateException("GGUF truncated at " + this.position);
      }
    }

    int readU32AsInt() {
      this.require(4);
      int value = this.map.getInt((int) this.position);
      this.position += 4;
      return value;
    }

    long readU64() {
      this.require(8);
      long value = this.map.getLong((int) this.position);
      this.position += 8;
      return value;
    }

    String readString() {
      long len = this.readU64();
      if (len < 0 || len > this.limits.maxGgufStringBytes()) {
        throw new IllegalStateException(
          "invalid GGUF string length " + len + " (maxGgufStringBytes="
            + this.limits.maxGgufStringBytes() + ")");
      }
      this.require((int) len);
      byte[] bytes = new byte[(int) len];
      int pos = (int) this.position;
      for (int i = 0; i < bytes.length; i++) {
        bytes[i] = this.map.get(pos + i);
      }
      this.position += len;
      return new String(bytes, UTF_8);
    }

    Object readValue(final int type) {
      return switch (type) {
        case 0 -> {
          this.require(1);
          byte v = this.map.get((int) this.position);
          this.position += 1;
          yield Byte.toUnsignedInt(v);
        }
        case 1 -> {
          this.require(1);
          byte v = this.map.get((int) this.position);
          this.position += 1;
          yield (int) v;
        }
        case 2 -> {
          this.require(2);
          int v = Short.toUnsignedInt(this.map.getShort((int) this.position));
          this.position += 2;
          yield v;
        }
        case 3 -> {
          this.require(2);
          short v = this.map.getShort((int) this.position);
          this.position += 2;
          yield (int) v;
        }
        case 4 -> this.readU32AsInt();
        case 5 -> {
          this.require(4);
          int v = this.map.getInt((int) this.position);
          this.position += 4;
          yield v;
        }
        case 6 -> {
          this.require(4);
          float v = this.map.getFloat((int) this.position);
          this.position += 4;
          yield v;
        }
        case 7 -> {
          this.require(1);
          byte v = this.map.get((int) this.position);
          this.position += 1;
          yield v != 0;
        }
        case 8 -> this.readString();
        case 9 -> this.readArray();
        case 10 -> this.readU64();
        case 11 -> {
          this.require(8);
          long v = this.map.getLong((int) this.position);
          this.position += 8;
          yield v;
        }
        case 12 -> {
          this.require(8);
          double v = this.map.getDouble((int) this.position);
          this.position += 8;
          yield v;
        }
        default -> throw new IllegalStateException("unsupported GGUF value type " + type);
      };
    }

    private List<Object> readArray() {
      int elemType = this.readU32AsInt();
      long len = this.readU64();
      if (len < 0 || len > 10_000_000) {
        throw new IllegalStateException("invalid GGUF array length " + len);
      }
      List<Object> values = new ArrayList<>((int) Math.min(len, 1_000_000));
      for (long i = 0; i < len; i++) {
        values.add(this.readValue(elemType));
      }
      return values;
    }
  }
}
