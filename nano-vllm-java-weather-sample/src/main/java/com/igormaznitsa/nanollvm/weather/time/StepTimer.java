package com.igormaznitsa.nanollvm.weather.time;

import static java.util.Objects.requireNonNull;

import java.util.Locale;
import java.util.function.Supplier;

/**
 * Wall-clock log for one sample run. Each measured step prints as it finishes.
 */
public final class StepTimer {

  private final long started = System.nanoTime();

  static String seconds(final long nanos) {
    return String.format(Locale.ROOT, "%.3f s", nanos / 1_000_000_000.0d);
  }

  public <T> T measure(final String name, final Supplier<T> action) {
    requireNonNull(action, "action");
    long began = System.nanoTime();
    T value = action.get();
    this.log(name, System.nanoTime() - began);
    return value;
  }

  public void measure(final String name, final Runnable action) {
    requireNonNull(action, "action");
    long began = System.nanoTime();
    action.run();
    this.log(name, System.nanoTime() - began);
  }

  public void log(final String name, final long nanos) {
    requireNonNull(name, "name");
    System.out.println(name + ": " + seconds(nanos));
  }

  public void logTotal() {
    System.out.println();
    this.log("total", System.nanoTime() - this.started);
  }
}
