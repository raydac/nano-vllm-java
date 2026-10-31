package com.igormaznitsa.nanollvm.samples.utils;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;
import one.profiler.AsyncProfiler;
import one.profiler.Counter;

/**
 * Optional async-profiler session around one timed stretch of {@link com.igormaznitsa.nanollvm.samples.Bench}.
 *
 * <p>Inactive unless {@code -Dnanollvm.profile} is set ({@code cpu}, {@code wall}, {@code alloc},
 * {@code lock}, or {@code itimer}). The native library is the one embedded in
 * {@code tools.profiler:async-profiler}. Reports are a flat histogram and collapsed stacks next to
 * {@code -Dnanollvm.profile.file} (a path prefix, default {@code bench-profile}).
 */
public final class BenchProfiler implements AutoCloseable {

  static final String PROP = "nanollvm.profile";
  static final String PROP_FILE = "nanollvm.profile.file";
  static final String PROP_INTERVAL = "nanollvm.profile.interval";
  static final long DEFAULT_INTERVAL_NS = 10_000_000L;

  private static final Set<String> EVENTS = Set.of("cpu", "wall", "alloc", "lock", "itimer");
  private static final int FLAT_METHODS = 80;

  private final AsyncProfiler profiler;
  private final Path collapsedFile;
  private final Path flatFile;

  private BenchProfiler(
    final AsyncProfiler profiler, final Path collapsedFile, final Path flatFile
  ) {
    this.profiler = profiler;
    this.collapsedFile = collapsedFile;
    this.flatFile = flatFile;
  }

  /**
   * Starts sampling when {@link #PROP} is set. Otherwise a closeable no-op.
   */
  public static BenchProfiler open() {
    String event = NanoLlvmProps.systemProperty(PROP);
    if (event == null) {
      return inactive();
    }
    return started(event);
  }

  static String startCommand(final String event, final long intervalNs) {
    if (intervalNs <= 0) {
      throw new IllegalArgumentException("nanollvm.profile.interval must be > 0");
    }
    return "start,event=%s,interval=%d".formatted(requireEvent(event), intervalNs);
  }

  static long requireInterval(final String raw) {
    if (raw == null) {
      return DEFAULT_INTERVAL_NS;
    }
    try {
      long interval = Long.parseLong(raw.strip());
      if (interval <= 0) {
        throw new IllegalArgumentException("nanollvm.profile.interval must be > 0");
      }
      return interval;
    } catch (NumberFormatException failed) {
      throw new IllegalArgumentException(
        "nanollvm.profile.interval must be a positive number of nanoseconds", failed);
    }
  }

  static Path withExtension(final Path base, final String extension) {
    Path fileName = requireNonNull(base, "base").getFileName();
    if (fileName == null) {
      throw new IllegalArgumentException("nanollvm.profile.file must be a file prefix");
    }
    return base.resolveSibling(fileName + extension);
  }

  private static BenchProfiler inactive() {
    return new BenchProfiler(null, null, null);
  }

  private static BenchProfiler started(final String event) {
    String command =
      startCommand(event, requireInterval(NanoLlvmProps.systemProperty(PROP_INTERVAL)));
    Path base = outputBase();
    AsyncProfiler profiler = loadProfiler();
    try {
      profiler.execute(command);
    } catch (RuntimeException | IOException failed) {
      throw new IllegalStateException(startFailure(event, failed), failed);
    }
    System.out.printf(Locale.ROOT, "async-profiler %s %s%n", profiler.getVersion(), command);
    return new BenchProfiler(
      profiler, withExtension(base, ".collapsed"), withExtension(base, ".flat"));
  }

  private static AsyncProfiler loadProfiler() {
    try {
      return AsyncProfiler.getInstance();
    } catch (UnsatisfiedLinkError | IllegalStateException failed) {
      throw new IllegalStateException(
        "async-profiler native library did not load. The samples module depends on tools.profiler:async-profiler, which extracts libasyncProfiler for this OS.",
        failed);
    }
  }

  private static Path outputBase() {
    String configured = NanoLlvmProps.systemProperty(PROP_FILE);
    if (configured == null) {
      return Path.of("bench-profile").toAbsolutePath();
    }
    return Path.of(configured).toAbsolutePath();
  }

  private static String requireEvent(final String event) {
    String normalized = requireNonNull(event, "event").strip().toLowerCase(Locale.ROOT);
    if (!EVENTS.contains(normalized)) {
      throw new IllegalArgumentException(
        "nanollvm.profile must be one of %s but was %s".formatted(EVENTS, event));
    }
    return normalized;
  }

  private static String startFailure(final String event, final Throwable failed) {
    return "async-profiler did not start for event %s (%s). CPU sampling needs perf events (kernel.perf_event_paranoid <= 1). -Dnanollvm.profile=itimer uses the interval timer instead."
      .formatted(event, failed.getMessage());
  }

  public boolean isActive() {
    return this.profiler != null;
  }

  @Override
  public void close() {
    if (this.profiler == null) {
      return;
    }
    try {
      String collapsed = this.profiler.dumpCollapsed(Counter.SAMPLES);
      String flat = this.profiler.dumpFlat(FLAT_METHODS);
      long samples = this.profiler.getSamples();
      this.write(this.collapsedFile, collapsed);
      this.write(this.flatFile, flat);
      System.out.printf(
        Locale.ROOT,
        "Profile: %d samples%n  %s%n  %s%n",
        samples,
        this.flatFile.toAbsolutePath(),
        this.collapsedFile.toAbsolutePath());
    } catch (IOException failed) {
      throw new IllegalStateException("Could not write the async-profiler report", failed);
    } finally {
      this.profiler.stop();
    }
  }

  private void write(final Path file, final String text) throws IOException {
    Path parent = file.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    String body = text.endsWith("\n") ? text : text + "\n";
    Files.writeString(file, body, StandardCharsets.UTF_8);
  }
}
