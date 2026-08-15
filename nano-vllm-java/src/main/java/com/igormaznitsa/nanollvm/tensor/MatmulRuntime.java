package com.igormaznitsa.nanollvm.tensor;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Dense matmul runtime for one {@code LLM}: parallelism cap plus an {@link ExecutorService}.
 *
 * <p>Build with {@link #builder()}, pass via step {@link com.igormaznitsa.nanollvm.internal.Context},
 * and {@link #close()} with the owning {@code LLM} (marks this runtime closed; does not shut down a
 * shared or caller-provided executor). Stateless float primitives stay on {@link VectorMath}.
 *
 * @see Ops#linear(Tensor, Tensor, Tensor)
 * @see VectorMath
 */
public final class MatmulRuntime implements AutoCloseable {

  private static final int TILE_N = 64;
  private static final int TILE_K = 256;
  private static final int MIN_PARALLEL_OUT = TILE_N * 2;
  private static final FloatKernels KERNELS = FloatKernels.get();
  private static final MatmulRuntime SEQUENTIAL = new MatmulRuntime(1, null, false);
  private static final AtomicReference<ExecutorService> SHARED_POOL = new AtomicReference<>();

  private final int cpuThreads;
  private final ExecutorService pool;
  private final boolean markClosedOnClose;
  private final AtomicBoolean closed = new AtomicBoolean();

  private MatmulRuntime(
    final int cpuThreads,
    final ExecutorService pool,
    final boolean markClosedOnClose
  ) {
    this.cpuThreads = cpuThreads;
    this.pool = pool;
    this.markClosedOnClose = markClosedOnClose;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Shared sequential runtime (no pool). Safe to reuse; {@link #close()} is a no-op.
   */
  public static MatmulRuntime sequential() {
    return SEQUENTIAL;
  }

  /**
   * Process-wide matmul pool, created on first parallel use ({@code availableProcessors} daemons).
   * Never shut down by the library.
   */
  static ExecutorService sharedExecutor() {
    ExecutorService existing = SHARED_POOL.get();
    if (existing != null) {
      return existing;
    }
    int threads = Math.max(2, Runtime.getRuntime().availableProcessors());
    ExecutorService created = Executors.newFixedThreadPool(threads, namedDaemonFactory());
    if (SHARED_POOL.compareAndSet(null, created)) {
      return created;
    }
    created.shutdownNow();
    return SHARED_POOL.get();
  }

  private static ThreadFactory namedDaemonFactory() {
    AtomicInteger seq = new AtomicInteger();
    return runnable -> {
      Thread thread = new Thread(runnable, "nanollvm-matmul-" + seq.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    };
  }

  public int cpuThreads() {
    return this.cpuThreads;
  }

  public String backendInfo() {
    String poolKind = this.pool == null
      ? "sequential"
      : (this.usesSharedPool() ? "shared-pool" : "custom-pool");
    return "%s, tileN=%d tileK=%d, cpuThreads=%d, %s".formatted(
      KERNELS.name(), TILE_N, TILE_K, this.cpuThreads, poolKind);
  }

  private boolean usesSharedPool() {
    ExecutorService shared = SHARED_POOL.get();
    return this.pool.equals(shared);
  }

  /**
   * Dense batched linear map: for each of {@code rows} inputs of width {@code in}, write
   * {@code out} outputs {@code y = x Wᵀ (+ bias)}.
   *
   * <p>Layouts match {@link VectorMath} / HF {@code [out, in]} weights. With
   * {@code cpuThreads > 1}, disjoint {@code out} ranges run on this runtime's executor.
   */
  public void linear(
    final float[] x, final int xOffset,
    final float[] w, final int wOffset,
    final float[] bias,
    final float[] y, final int yOffset,
    final int rows, final int in, final int out
  ) {
    this.requireOpen();
    if (rows == 1) {
      this.linearDecode1(x, xOffset, w, wOffset, bias, y, yOffset, in, out);
      return;
    }
    if (this.pool == null || this.cpuThreads <= 1 || out < MIN_PARALLEL_OUT) {
      this.linearRange(x, xOffset, w, wOffset, bias, y, yOffset, rows, in, out, 0, out);
      return;
    }

    int workers = Math.min(this.cpuThreads, (out + TILE_N - 1) / TILE_N);
    if (workers <= 1) {
      this.linearRange(x, xOffset, w, wOffset, bias, y, yOffset, rows, in, out, 0, out);
      return;
    }

    int chunk = (out + workers - 1) / workers;
    List<Callable<Void>> tasks = new ArrayList<>(workers);
    for (int worker = 0; worker < workers; worker++) {
      int out0 = worker * chunk;
      int out1 = Math.min(out, out0 + chunk);
      if (out0 >= out1) {
        break;
      }
      tasks.add(() -> {
        this.linearRange(x, xOffset, w, wOffset, bias, y, yOffset, rows, in, out, out0, out1);
        return null;
      });
    }
    this.awaitAll(tasks);
  }

  /**
   * Dense decode path ({@code rows == 1}): one activation vector against {@code [out, in]} weights.
   */
  public void linearDecode1(
    final float[] x, final int xOffset,
    final float[] w, final int wOffset,
    final float[] bias,
    final float[] y, final int yOffset,
    final int in, final int out
  ) {
    this.requireOpen();
    if (this.pool == null || this.cpuThreads <= 1 || out < MIN_PARALLEL_OUT) {
      this.decode1Range(x, xOffset, w, wOffset, bias, y, yOffset, in, 0, out);
      return;
    }

    int workers = Math.min(this.cpuThreads, (out + TILE_N - 1) / TILE_N);
    if (workers <= 1) {
      this.decode1Range(x, xOffset, w, wOffset, bias, y, yOffset, in, 0, out);
      return;
    }

    int chunk = (out + workers - 1) / workers;
    List<Callable<Void>> tasks = new ArrayList<>(workers);
    for (int worker = 0; worker < workers; worker++) {
      int out0 = worker * chunk;
      int out1 = Math.min(out, out0 + chunk);
      if (out0 >= out1) {
        break;
      }
      tasks.add(() -> {
        this.decode1Range(x, xOffset, w, wOffset, bias, y, yOffset, in, out0, out1);
        return null;
      });
    }
    this.awaitAll(tasks);
  }

  /**
   * Packed GGUF linear: dequant one weight row at a time, then dot. Parallelizes disjoint
   * {@code out} ranges when {@code cpuThreads > 1} (each worker owns a row scratch buffer).
   */
  public void linearPacked(
    final float[] x, final int xOffset,
    final PackedWeight weight,
    final float[] bias,
    final float[] y, final int yOffset,
    final int rows, final int in, final int out
  ) {
    this.linearPackedRows(x, xOffset, weight::dequantizeRow, bias, y, yOffset, rows, in, out);
  }

  /**
   * Packed linear with a caller-supplied row dequant (type-specialized kernels use this).
   */
  public void linearPackedRows(
    final float[] x, final int xOffset,
    final PackedRowDequant dequant,
    final float[] bias,
    final float[] y, final int yOffset,
    final int rows, final int in, final int out
  ) {
    this.requireOpen();
    requireNonNull(dequant, "dequant");
    if (this.pool == null || this.cpuThreads <= 1 || out < MIN_PARALLEL_OUT) {
      this.packedLinearRange(x, xOffset, dequant, bias, y, yOffset, rows, in, out, 0, out);
      return;
    }

    int workers = Math.min(this.cpuThreads, (out + TILE_N - 1) / TILE_N);
    if (workers <= 1) {
      this.packedLinearRange(x, xOffset, dequant, bias, y, yOffset, rows, in, out, 0, out);
      return;
    }

    int chunk = (out + workers - 1) / workers;
    List<Callable<Void>> tasks = new ArrayList<>(workers);
    for (int worker = 0; worker < workers; worker++) {
      int out0 = worker * chunk;
      int out1 = Math.min(out, out0 + chunk);
      if (out0 >= out1) {
        break;
      }
      tasks.add(() -> {
        this.packedLinearRange(x, xOffset, dequant, bias, y, yOffset, rows, in, out, out0, out1);
        return null;
      });
    }
    this.awaitAll(tasks);
  }

  private void linearRange(
    final float[] x, final int xOffset,
    final float[] w, final int wOffset,
    final float[] bias,
    final float[] y, final int yOffset,
    final int rows, final int in, final int out,
    final int out0, final int out1
  ) {
    for (int r = 0; r < rows; r++) {
      KERNELS.gemv(
        x, xOffset + r * in, w, wOffset, bias, y, yOffset + r * out, in, out0, out1);
    }
  }

  @Override
  public void close() {
    if (this.markClosedOnClose) {
      this.closed.set(true);
    }
  }

  private void requireOpen() {
    if (this.markClosedOnClose && this.closed.get()) {
      throw new IllegalStateException("MatmulRuntime is closed");
    }
  }

  private void decode1Range(
    final float[] x, final int xOffset,
    final float[] w, final int wOffset,
    final float[] bias,
    final float[] y, final int yOffset,
    final int in,
    final int out0, final int out1
  ) {
    KERNELS.gemv(x, xOffset, w, wOffset, bias, y, yOffset, in, out0, out1);
  }

  private void packedLinearRange(
    final float[] x, final int xOffset,
    final PackedRowDequant dequant,
    final float[] bias,
    final float[] y, final int yOffset,
    final int rows, final int in, final int out,
    final int out0, final int out1
  ) {
    float[] row = new float[in];
    for (int o = out0; o < out1; o++) {
      dequant.dequantizeRow(o, row);
      float b = bias != null ? bias[o] : 0f;
      for (int r = 0; r < rows; r++) {
        y[yOffset + r * out + o] = b + KERNELS.dot(x, xOffset + r * in, row, 0, in);
      }
    }
  }

  private void awaitAll(final List<Callable<Void>> tasks) {
    try {
      List<Future<Void>> futures = this.pool.invokeAll(tasks);
      for (Future<Void> future : futures) {
        future.get();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("CPU matmul interrupted", e);
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("CPU matmul failed", cause);
    }
  }

  @FunctionalInterface
  public interface PackedRowDequant {
    void dequantizeRow(int row, float[] dst);
  }

  public static final class Builder {

    private int cpuThreads = 1;
    private ExecutorService executor;

    private Builder() {
    }

    /**
     * Max parallel matmul chunks for this runtime ({@code 1} = sequential, no executor use).
     */
    public Builder cpuThreads(final int value) {
      if (value < 1) {
        throw new IllegalArgumentException("cpuThreads must be >= 1, got " + value);
      }
      this.cpuThreads = value;
      return this;
    }

    public Builder allCpuThreads() {
      return this.cpuThreads(Runtime.getRuntime().availableProcessors());
    }

    public Builder disableMultiCpu() {
      return this.cpuThreads(1);
    }

    /**
     * Executor for parallel matmul chunks. Not shut down by {@link MatmulRuntime#close()}.
     * When omitted and {@code cpuThreads > 1}, the lazily created shared pool is used.
     * Ignored when {@code cpuThreads == 1} ({@link #disableMultiCpu()} / sequential).
     */
    public Builder executor(final ExecutorService executor) {
      this.executor = requireNonNull(executor, "executor");
      return this;
    }

    public MatmulRuntime build() {
      if (this.cpuThreads == 1) {
        return sequential();
      }
      ExecutorService pool = this.executor != null ? this.executor : sharedExecutor();
      return new MatmulRuntime(this.cpuThreads, pool, true);
    }
  }
}
