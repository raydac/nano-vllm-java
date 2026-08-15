package com.igormaznitsa.nanollvm.models.llmcontainer;

import static java.util.Locale.ROOT;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;

/**
 * In-place percent / ETA bar for weight-file decode (safetensors, GGUF, ONNX).
 *
 * @since 1.1.0
 */
public final class LoadProgress {
  private final String label;
  private final int total;
  private final LlmListener io;
  private final long startNanos = System.nanoTime();
  private int current;
  private String detail = "";
  private boolean finished;

  /**
   * Starts a bar for {@code total} steps. {@code io} may be {@code null} (silent).
   *
   * @since 1.1.0
   */
  public LoadProgress(final String label, final int total, final LlmListener io) {
    this.label = label;
    this.total = Math.max(1, total);
    this.io = io;
    this.render();
  }

  private static String formatSeconds(final double seconds) {
    if (seconds < 60) {
      return String.format(ROOT, "%.0fs", seconds);
    }
    return String.format(ROOT, "%dm%02ds", (int) (seconds / 60), (int) (seconds % 60));
  }

  /**
   * Advances one step and redraws.
   *
   * @since 1.1.0
   */
  public void step(final String detail) {
    this.current = Math.min(this.current + 1, this.total);
    this.detail = detail == null ? "" : detail;
    this.render();
  }

  /**
   * Completes the bar with a final status line. Idempotent.
   *
   * @since 1.1.0
   */
  public void finish(final String message) {
    if (this.finished) {
      return;
    }
    this.finished = true;
    if (LlmListeners.isSilent(this.io)) {
      return;
    }
    double seconds = (System.nanoTime() - this.startNanos) / 1e9;
    LlmListeners.infof(this.io, null, "\r%s: done in %.1fs%s%n",
      this.label, seconds, message == null || message.isBlank() ? "" : " — " + message);
  }

  private void render() {
    if (this.finished || LlmListeners.isSilent(this.io)) {
      return;
    }
    double fraction = (double) this.current / this.total;
    int width = 24;
    int filled = (int) Math.round(fraction * width);
    String bar = "=".repeat(Math.max(0, filled)) + " ".repeat(Math.max(0, width - filled));
    double elapsed = (System.nanoTime() - this.startNanos) / 1e9;
    String eta = this.current <= 0 || fraction <= 0
      ? "--"
      : formatSeconds(elapsed * (1.0 - fraction) / fraction);
    String shortDetail = this.detail.length() <= 48
      ? this.detail
      : "…" + this.detail.substring(this.detail.length() - 47);
    LlmListeners.infof(this.io, null, "\r%s: [%s] %3.0f%% (%d/%d) ETA %s  %s   ",
      this.label, bar, fraction * 100.0, this.current, this.total, eta, shortDetail);
  }
}
