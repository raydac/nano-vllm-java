package io.nanovllm.utils;

import io.nanovllm.models.CausalLM;
import io.nanovllm.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class ModelLoader {

  private ModelLoader() {
  }

  public static void loadModel(CausalLM model, Path modelDir) throws IOException {
    Map<String, Object[]> packed = model.packedModulesMapping();
    List<Path> files = SafetensorsReader.listSafetensors(modelDir);
    if (files.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors files in " + modelDir);
    }

    long fileBytes = 0L;
    for (Path file : files) {
      fileBytes += Files.size(file);
    }
    System.err.printf(Locale.ROOT, "Loading weights from %s (%.2f GiB, %d file%s)%n",
        modelDir,
        fileBytes / (1024.0 * 1024.0 * 1024.0),
        files.size(),
        files.size() == 1 ? "" : "s");

    List<PlannedTensor> plan = new ArrayList<>();
    for (Path file : files) {
      try (SafetensorsReader probe = SafetensorsReader.open(file)) {
        for (String weightName : probe.keys()) {
          ResolvedParam resolved = resolveParam(weightName, model, packed);
          if (resolved == null) {
            continue;
          }
          plan.add(new PlannedTensor(file, weightName, resolved.paramName(), resolved.shardId(),
              probe.byteSize(weightName)));
        }
      }
    }
    if (plan.isEmpty()) {
      throw new IllegalStateException("no matching weight tensors found in " + modelDir);
    }

    Progress progress = new Progress("Weights", plan.size());
    long loadedBytes = 0L;
    Path currentFile = null;
    SafetensorsReader reader = null;
    try {
      for (PlannedTensor item : plan) {
        if (reader == null || !item.file().equals(currentFile)) {
          if (reader != null) {
            reader.close();
          }
          currentFile = item.file();
          reader = SafetensorsReader.open(currentFile);
        }
        Tensor tensor = reader.getTensor(item.weightName());
        model.getParameter(item.paramName()).load(tensor, item.shardId());
        loadedBytes += item.byteSize();
        progress.step("%s | %s (%.0f MiB)".formatted(
            item.file().getFileName(),
            item.weightName(),
            loadedBytes / (1024.0 * 1024.0)));
      }
      progress.finish("%.0f MiB loaded".formatted(loadedBytes / (1024.0 * 1024.0)));
    } catch (RuntimeException | IOException e) {
      progress.finish("failed");
      throw e;
    } finally {
      if (reader != null) {
        reader.close();
      }
    }
  }

  private static ResolvedParam resolveParam(
      String weightName,
      CausalLM model,
      Map<String, Object[]> packed
  ) {
    for (var e : packed.entrySet()) {
      String key = e.getKey();
      if (weightName.contains(key)) {
        Object[] mapping = e.getValue();
        String paramName = weightName.replace(key, (String) mapping[0]);
        if (!model.hasParameter(paramName)) {
          return null;
        }
        return new ResolvedParam(paramName, mapping[1]);
      }
    }
    if (!model.hasParameter(weightName)) {
      return null;
    }
    return new ResolvedParam(weightName, null);
  }

  private record ResolvedParam(String paramName, Object shardId) {
  }

  private record PlannedTensor(
      Path file,
      String weightName,
      String paramName,
      Object shardId,
      long byteSize
  ) {
  }

  private static final class Progress {
    private final String label;
    private final int total;
    private final long startNanos = System.nanoTime();
    private int current;
    private String detail = "";
    private boolean finished;

    Progress(String label, int total) {
      this.label = label;
      this.total = Math.max(1, total);
      this.render();
    }

    private static String formatSeconds(double seconds) {
      if (seconds < 60) {
        return String.format(Locale.ROOT, "%.0fs", seconds);
      }
      return String.format(Locale.ROOT, "%dm%02ds", (int) (seconds / 60), (int) (seconds % 60));
    }

    void step(String detail) {
      this.current = Math.min(this.current + 1, this.total);
      this.detail = detail == null ? "" : detail;
      this.render();
    }

    void finish(String message) {
      if (this.finished) {
        return;
      }
      this.finished = true;
      double seconds = (System.nanoTime() - this.startNanos) / 1e9;
      System.err.printf(Locale.ROOT, "\r%s: done in %.1fs%s%n",
          this.label, seconds, message == null || message.isBlank() ? "" : " — " + message);
      System.err.flush();
    }

    private void render() {
      if (this.finished) {
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
      System.err.printf(Locale.ROOT, "\r%s: [%s] %3.0f%% (%d/%d) ETA %s  %s   ",
          this.label, bar, fraction * 100.0, this.current, this.total, eta, shortDetail);
      System.err.flush();
    }
  }
}
