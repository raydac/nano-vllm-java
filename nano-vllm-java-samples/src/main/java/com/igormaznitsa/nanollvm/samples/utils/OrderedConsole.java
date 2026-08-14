package com.igormaznitsa.nanollvm.samples.utils;

import static java.util.Objects.requireNonNull;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Demo console with two ordered sinks: stdout for the visible answer / prompts / turn stats, and an
 * info sink (typically {@link System#err}) for load status, thinking, advisor notes, and debug.
 */
public final class OrderedConsole {

  private final PrintStream out;
  private final PrintStream info;
  private final ConcurrentLinkedQueue<Chunk> outQueue = new ConcurrentLinkedQueue<>();
  private final ConcurrentLinkedQueue<Chunk> infoQueue = new ConcurrentLinkedQueue<>();
  private final AtomicLong seq = new AtomicLong();
  private final Object drainLock = new Object();
  private final PrintStream outStream;
  private final PrintStream infoStream;

  public OrderedConsole(final PrintStream out) {
    this(out, out);
  }

  public OrderedConsole(final PrintStream out, final PrintStream info) {
    this.out = requireNonNull(out, "out");
    this.info = requireNonNull(info, "info");
    this.outStream = new PrintStream(new EnqueueStream(false), false, StandardCharsets.UTF_8);
    this.infoStream = new PrintStream(new EnqueueStream(true), false, StandardCharsets.UTF_8);
  }

  /**
   * Stdout queue for assistant answers, prompts, and turn stats.
   */
  public PrintStream stream() {
    return this.outStream;
  }

  /**
   * Info/stderr queue for thinking, advisor notes, debug, and status lines.
   */
  public PrintStream infoStream() {
    return this.infoStream;
  }

  public void print(final String text) {
    this.enqueue(this.outQueue, text == null ? "" : text);
    this.drain();
  }

  public void println(final String text) {
    this.print((text == null ? "" : text) + "\n");
  }

  public void println() {
    this.print("\n");
  }

  @SuppressWarnings("AnnotateFormatMethod")
  public void printf(final Locale locale, final String format, final Object... args) {
    this.print(String.format(locale, format, args));
  }

  public void printInfo(final String text) {
    this.enqueue(this.infoQueue, text == null ? "" : text);
    this.drain();
  }

  public void printlnInfo(final String text) {
    this.printInfo((text == null ? "" : text) + "\n");
  }

  private void enqueue(final ConcurrentLinkedQueue<Chunk> queue, final String text) {
    if (text.isEmpty()) {
      return;
    }
    queue.offer(new Chunk(System.currentTimeMillis(), this.seq.getAndIncrement(), text));
  }

  private void drain() {
    synchronized (this.drainLock) {
      this.drainQueue(this.infoQueue, this.info);
      this.drainQueue(this.outQueue, this.out);
    }
  }

  private void drainQueue(final ConcurrentLinkedQueue<Chunk> queue, final PrintStream sink) {
    List<Chunk> batch = new ArrayList<>();
    for (Chunk chunk; (chunk = queue.poll()) != null; ) {
      batch.add(chunk);
    }
    if (batch.isEmpty()) {
      return;
    }
    batch.sort(Comparator.comparingLong(Chunk::millis).thenComparingLong(Chunk::seq));
    for (Chunk chunk : batch) {
      sink.print(chunk.text());
    }
    sink.flush();
  }

  private record Chunk(long millis, long seq, String text) {
  }

  private final class EnqueueStream extends OutputStream {
    private final boolean toInfo;
    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    EnqueueStream(final boolean toInfo) {
      this.toInfo = toInfo;
    }

    @Override
    public void write(final int b) {
      synchronized (this) {
        this.buffer.write(b);
      }
    }

    @Override
    public void write(final byte[] bytes, final int off, final int len) {
      synchronized (this) {
        this.buffer.write(bytes, off, len);
      }
    }

    @Override
    public void flush() {
      final String text;
      synchronized (this) {
        if (this.buffer.size() == 0) {
          return;
        }
        text = this.buffer.toString(StandardCharsets.UTF_8);
        this.buffer.reset();
      }
      ConcurrentLinkedQueue<Chunk> queue =
        this.toInfo ? OrderedConsole.this.infoQueue : OrderedConsole.this.outQueue;
      OrderedConsole.this.enqueue(queue, text);
      OrderedConsole.this.drain();
    }
  }
}
