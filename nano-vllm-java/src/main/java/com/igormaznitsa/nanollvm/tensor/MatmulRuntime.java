package com.igormaznitsa.nanollvm.tensor;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
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
 * <p>Build with {@link #builder()}, pass via step {@link com.igormaznitsa.nanollvm.internal.Context}.
 * {@link #parallelRanges(int, RangeTask)} splits independent index ranges across the same pool
 * (attention heads, RoPE tokens, embedding rows, BERT queries). Nested calls from a pool worker
 * run sequentially so a fixed thread pool cannot deadlock on {@code invokeAll}.
 *
 * @see Ops#linear(Tensor, Tensor, Tensor)
 * @see VectorMath
 */
public final class MatmulRuntime implements AutoCloseable {

  private static final int TILE_N = 64;
  private static final int TILE_K = 256;
  private static final int MIN_PARALLEL_OUT = TILE_N * 2;
  private static final FloatKernels KERNELS = FloatKernels.get();
  private static final MatmulRuntime SEQUENTIAL = new MatmulRuntime(1, null, false, false, false);
  private static final AtomicReference<SharedPool> SHARED = new AtomicReference<>();
  private static final Set<Thread> POOL_WORKERS = ConcurrentHashMap.newKeySet();
  private final AtomicBoolean checkoutReleased = new AtomicBoolean();
  private final AtomicBoolean closed = new AtomicBoolean();

  private final int cpuThreads;
  private final ExecutorService pool;
  private final boolean markClosedOnClose;
  private final boolean sharedCheckout;
  private final boolean shutdownPoolOnClose;

  /**
   * Process-wide matmul pool, created on first parallel use ({@code availableProcessors} daemons).
   * Shut down when the last runtime that checked it out is {@link #close() closed}.
   */
  private static ExecutorService checkoutSharedPool() {
    while (true) {
      SharedPool current = SHARED.get();
      if (current == null) {
        ExecutorService created = newSharedExecutor();
        if (SHARED.compareAndSet(null, new SharedPool(created, 1))) {
          return created;
        }
        created.shutdownNow();
        continue;
      }
      SharedPool leased = new SharedPool(current.executor(), current.checkouts() + 1);
      if (SHARED.compareAndSet(current, leased)) {
        return current.executor();
      }
    }
  }

  private MatmulRuntime(
    final int cpuThreads,
    final ExecutorService pool,
    final boolean markClosedOnClose,
    final boolean sharedCheckout,
    final boolean shutdownPoolOnClose
  ) {
    this.cpuThreads = cpuThreads;
    this.pool = pool;
    this.markClosedOnClose = markClosedOnClose;
    this.sharedCheckout = sharedCheckout;
    this.shutdownPoolOnClose = shutdownPoolOnClose;
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

  private static void releaseSharedCheckout() {
    while (true) {
      SharedPool current = SHARED.get();
      if (current == null) {
        return;
      }
      if (current.checkouts() <= 1) {
        if (SHARED.compareAndSet(current, null)) {
          current.executor().shutdownNow();
          return;
        }
        continue;
      }
      SharedPool remaining = new SharedPool(current.executor(), current.checkouts() - 1);
      if (SHARED.compareAndSet(current, remaining)) {
        return;
      }
    }
  }

  private static ExecutorService newSharedExecutor() {
    return Executors.newFixedThreadPool(
      Math.max(2, Runtime.getRuntime().availableProcessors()), namedDaemonFactory());
  }

  private static ExecutorService newOwnedExecutor(final int cpuThreads) {
    AtomicInteger seq = new AtomicInteger();
    ThreadFactory factory = runnable -> {
      Thread thread = new Thread(runnable, "nanollvm-matmul-owned-" + seq.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    };
    return Executors.newFixedThreadPool(cpuThreads, factory);
  }

  @Override
  public void close() {
    if (this.markClosedOnClose) {
      this.closed.set(true);
    }
    if (this.sharedCheckout) {
      if (this.checkoutReleased.compareAndSet(false, true)) {
        releaseSharedCheckout();
      }
      return;
    }
    if (!this.shutdownPoolOnClose || this.pool == null) {
      return;
    }
    if (!this.checkoutReleased.compareAndSet(false, true)) {
      return;
    }
    this.pool.shutdownNow();
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
    String poolKind = this.poolKindLabel();
    return "%s, tileN=%d tileK=%d, cpuThreads=%d, %s".formatted(
      KERNELS.name(), TILE_N, TILE_K, this.cpuThreads, poolKind);
  }

  private String poolKindLabel() {
    if (this.pool == null) {
      return "sequential";
    }
    if (this.sharedCheckout) {
      return "shared-pool";
    }
    if (this.shutdownPoolOnClose) {
      return "owned-pool";
    }
    return "custom-pool";
  }

  private boolean parallelEnabled() {
    return this.pool != null && this.cpuThreads > 1 &&
      !POOL_WORKERS.contains(Thread.currentThread());
  }

  /**
   * Splits {@code [0, work)} into disjoint ranges and runs {@code task} on each.
   * Sequential when this runtime has no pool, a single worker, {@code work <= 1}, or the
   * caller is already a pool worker (nested {@code invokeAll} would deadlock a fixed pool).
   *
   * @param work exclusive end of the index space; {@code <= 0} is a no-op
   * @param task receives {@code [start, end)} that this worker owns; must not be {@code null}
   */
  public void parallelRanges(final int work, final RangeTask task) {
    this.requireOpen();
    requireNonNull(task, "task");
    if (work <= 0) {
      return;
    }
    if (!this.parallelEnabled() || work == 1) {
      task.run(0, work);
      return;
    }

    int workers = Math.min(this.cpuThreads, work);
    if (workers <= 1) {
      task.run(0, work);
      return;
    }

    int chunk = (work + workers - 1) / workers;
    List<Callable<Void>> tasks = new ArrayList<>(workers);
    for (int worker = 0; worker < workers; worker++) {
      int start = worker * chunk;
      int end = Math.min(work, start + chunk);
      if (start >= end) {
        break;
      }
      tasks.add(() -> {
        task.run(start, end);
        return null;
      });
    }
    this.awaitAll(tasks);
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
    if (!this.parallelEnabled() || out < MIN_PARALLEL_OUT) {
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
    if (!this.parallelEnabled() || out < MIN_PARALLEL_OUT) {
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
    if (!this.parallelEnabled() || out < MIN_PARALLEL_OUT) {
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

  private void awaitAll(final List<Callable<Void>> tasks) {
    List<Callable<Void>> wrapped = new ArrayList<>(tasks.size());
    for (Callable<Void> task : tasks) {
      wrapped.add(() -> {
        Thread worker = Thread.currentThread();
        POOL_WORKERS.add(worker);
        try {
          return task.call();
        } finally {
          POOL_WORKERS.remove(worker);
        }
      });
    }
    try {
      List<Future<Void>> futures = this.pool.invokeAll(wrapped);
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

  private record SharedPool(ExecutorService executor, int checkouts) {
  }

  private void requireOpen() {
    if (this.closed.get()) {
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

  /**
   * Independent work over a half-open index range. Implementations must not share mutable
   * scratch with other ranges; writes to disjoint output slices are allowed.
   */
  @FunctionalInterface
  public interface RangeTask {
    void run(int startInclusive, int endExclusive);
  }

  @FunctionalInterface
  public interface PackedRowDequant {
    void dequantizeRow(int row, float[] dst);
  }

  public static final class Builder {

    private int cpuThreads = 1;
    private ExecutorService executor;
    private boolean dedicatedPool;

    private Builder() {
    }

    /**
     * Max parallel kernel chunks for this runtime ({@code 1} = sequential, no executor use).
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
     * When omitted and {@code cpuThreads > 1}, the lazily created shared pool is used and is
     * shut down when the last runtime using it closes, unless {@link #dedicatedPool()} was set.
     * Ignored when {@code cpuThreads == 1} ({@link #disableMultiCpu()} / sequential).
     * Cannot be combined with {@link #dedicatedPool()}.
     */
    public Builder executor(final ExecutorService executor) {
      this.executor = requireNonNull(executor, "executor");
      return this;
    }

    /**
     * Creates a bounded pool of {@link #cpuThreads(int)} daemon workers owned by this runtime.
     * {@link MatmulRuntime#close()} shuts it down. Does not join the process-wide shared pool.
     * Ignored when {@code cpuThreads == 1}. Cannot be combined with {@link #executor}.
     *
     * @return {@code this}
     */
    public Builder dedicatedPool() {
      this.dedicatedPool = true;
      return this;
    }

    public MatmulRuntime build() {
      if (this.cpuThreads == 1) {
        return sequential();
      }
      if (this.executor != null && this.dedicatedPool) {
        throw new IllegalStateException("cannot combine executor() with dedicatedPool()");
      }
      if (this.executor != null) {
        return new MatmulRuntime(this.cpuThreads, this.executor, true, false, false);
      }
      if (this.dedicatedPool) {
        return new MatmulRuntime(
          this.cpuThreads, newOwnedExecutor(this.cpuThreads), true, false, true);
      }
      return new MatmulRuntime(this.cpuThreads, checkoutSharedPool(), true, true, false);
    }
  }
}
