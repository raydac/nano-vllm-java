package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class KernelChatBenchTest {

  @Test
  void skippedModesOmitsAvailableBackends() {
    assertEquals(List.of(), KernelChatBench.skippedModes(List.of("scalar", "vector", "tornado")));
    assertEquals(List.of("tornado"), KernelChatBench.skippedModes(List.of("scalar", "vector")));
    assertEquals(
      List.of("vector", "tornado"),
      KernelChatBench.skippedModes(List.of("scalar")));
  }

  @Test
  void coordinatorModesAlwaysIncludeVectorEvenIfParentLacksIt() {
    assertEquals(
      List.of("scalar", "vector"),
      KernelChatBench.coordinatorModes(List.of("scalar")));
    assertEquals(
      List.of("scalar", "vector", "tornado"),
      KernelChatBench.coordinatorModes(List.of("scalar", "tornado")));
    assertEquals(
      List.of("scalar", "vector", "tornado"),
      KernelChatBench.coordinatorModes(List.of("scalar", "vector", "tornado")));
  }

  @Test
  void resultLineRoundTrip() {
    KernelChatBench.Result original = new KernelChatBench.Result(
      "vector",
      "Vector API (SIMD)",
      18,
      64,
      2_500_000_000L,
      25.6d);
    String line = KernelChatBench.formatResultLine(original);
    assertTrue(line.startsWith(KernelChatBench.RESULT_PREFIX + "\t"));

    Optional<KernelChatBench.Result> parsed = KernelChatBench.parseResultLine(line);
    assertTrue(parsed.isPresent());
    KernelChatBench.Result result = parsed.orElseThrow();
    assertEquals(original.mode(), result.mode());
    assertEquals(original.label(), result.label());
    assertEquals(original.promptTokens(), result.promptTokens());
    assertEquals(original.completionTokens(), result.completionTokens());
    assertEquals(original.elapsedNanos(), result.elapsedNanos());
    assertEquals(original.completionTokensPerSecond(), result.completionTokensPerSecond(), 1e-6);
  }

  @Test
  void parseResultLineIgnoresNoise() {
    assertTrue(KernelChatBench.parseResultLine("Loading model from x").isEmpty());
    assertTrue(KernelChatBench.parseResultLine("KERNEL_RESULT\ttoo\tfew").isEmpty());
    assertTrue(KernelChatBench.parseResultLine(null).isEmpty());
  }

  @Test
  void consoleTableIncludesSkippedTornadoAndTokPerSec() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(
      new String[] {"--svg", "-", "--max-tokens", "64", "--runs", "2"});
    List<KernelChatBench.Result> results = List.of(
      new KernelChatBench.Result("scalar", "Scalar Java", 18, 64, 10_000_000_000L, 6.4d),
      new KernelChatBench.Result("vector", "Vector API (SIMD)", 18, 64, 4_000_000_000L, 16.0d));
    String table = KernelChatBench.formatConsoleTable(
      Path.of("models/Gemma3-270M"),
      options,
      List.of("tornado"),
      results);

    assertTrue(table.contains("Gemma3-270M"));
    assertTrue(table.contains("skipped: tornado (TornadoVM unavailable (no SDK/device))"));
    assertTrue(table.contains("Scalar Java"));
    assertTrue(table.contains("Vector API (SIMD)"));
    assertTrue(table.contains("6.40"));
    assertTrue(table.contains("16.00"));
    assertTrue(table.contains("2.50x"));
  }

  @Test
  void parseOptionsReadsSvgDashAndModel() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(
      new String[] {"models/Gemma3-270M", "--svg", "-", "--max-tokens", "32", "--runs", "3"});
    assertFalse(options.worker());
    assertEquals(Path.of("models/Gemma3-270M"), options.modelPath());
    assertNull(options.svgPath());
    assertEquals(32, options.maxTokens());
    assertEquals(3, options.runs());
  }

  @Test
  void parseOptionsWorkerFlag() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(
      new String[] {"--worker", "--max-tokens", "8", "models/Qwen3-0.6B"});
    assertTrue(options.worker());
    assertEquals(Path.of("models/Qwen3-0.6B"), options.modelPath());
    assertEquals(8, options.maxTokens());
  }
}
