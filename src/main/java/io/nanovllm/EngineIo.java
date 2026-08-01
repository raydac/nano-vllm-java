package io.nanovllm;

import static java.util.Objects.requireNonNull;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Status / progress streams for the engine. Not part of {@link Config}.
 * Library default is {@link #silent()}; CLI tools use {@link #system()}.
 */
public final class EngineIo {

  private final PrintStream out;
  private final PrintStream err;
  private final boolean silent;

  private EngineIo(PrintStream out, PrintStream err, boolean silent) {
    this.out = requireNonNull(out, "out");
    this.err = requireNonNull(err, "err");
    this.silent = silent;
  }

  public static EngineIo silent() {
    PrintStream sink =
        new PrintStream(OutputStream.nullOutputStream(), false, StandardCharsets.UTF_8);
    return new EngineIo(sink, sink, true);
  }

  public static EngineIo system() {
    return new EngineIo(System.out, System.err, false);
  }

  public static EngineIo of(PrintStream out, PrintStream err) {
    return new EngineIo(out, err, false);
  }

  public PrintStream out() {
    return this.out;
  }

  public PrintStream err() {
    return this.err;
  }

  public boolean isSilent() {
    return this.silent;
  }

  public void info(String message) {
    this.err.println(message);
  }

  public void infof(String format, Object... args) {
    this.err.printf(Locale.ROOT, format, args);
  }

  public void progressf(String format, Object... args) {
    this.out.printf(Locale.ROOT, format, args);
  }
}
