package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.io.File;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
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
  void coordinatorModesIncludeTornadoWhenLauncherExists() {
    assertEquals(
      List.of("scalar", "vector"),
      KernelChatBench.coordinatorModes(List.of("scalar"), false));
    assertEquals(
      List.of("scalar", "vector", "tornado"),
      KernelChatBench.coordinatorModes(List.of("scalar"), true));
    assertEquals(
      List.of("scalar", "vector", "tornado"),
      KernelChatBench.coordinatorModes(List.of("scalar", "tornado"), false));
  }

  @Test
  void tornadoChildUsesLauncherRatherThanPlainJava() {
    Path tornado = Path.of("/opt/tornado");
    List<String> command = KernelChatBench.tornadoChildCommand(
      tornado,
      "classes",
      "/models/Gemma 3",
      8,
      2);
    assertEquals(tornado.toString(), command.getFirst());
    assertTrue(command.contains("--classpath"));
    assertTrue(command.contains(KernelChatBench.shellQuote("classes")));
    assertTrue(command.stream().anyMatch(arg -> arg.contains("-Dnanollvm.kernels=tornado")));
    assertTrue(command.stream().anyMatch(arg ->
        arg.contains("--worker") && arg.contains(KernelChatBench.shellQuote("/models/Gemma 3"))));
    assertEquals(KernelChatBench.class.getName(), command.getLast());
  }

  @Test
  void tornadoClasspathIsQuotedForCmdSeparator() {
    String windowsCp = "C:\\samples\\classes;C:\\nano-vllm-java\\classes";
    List<String> command = KernelChatBench.tornadoChildCommand(
        Path.of("C:\\tvm\\bin\\tornado.exe"),
        windowsCp,
        "C:\\models\\Gemma",
        8,
        2);
    assertEquals(
        KernelChatBench.shellQuote(windowsCp, System.getProperty("os.name", "")),
        command.get(command.indexOf("--classpath") + 1));
  }

  @Test
  void shellQuoteUsesCmdRulesOnWindows() {
    assertEquals("\"C:\\models\\Gemma 3\"",
        KernelChatBench.shellQuote("C:\\models\\Gemma 3", "Windows 11"));
    assertEquals("\"C:\\models\\say \"\"hi\"\"\"",
        KernelChatBench.shellQuote("C:\\models\\say \"hi\"", "Windows 11"));
    assertEquals("'/models/Gemma 3'", KernelChatBench.shellQuote("/models/Gemma 3", "Linux"));
  }

  @Test
  void tornadoExecutablePrefersPathThenHome() throws Exception {
    String launcher = KernelChatBench.tornadoLauncherNames(System.getProperty("os.name", ""))
        .getFirst();
    Path dir = Files.createTempDirectory("tornado-path");
    Path home = Files.createTempDirectory("tornado-home");
    try {
      Path onPath = dir.resolve(launcher);
      Files.writeString(onPath, "#!/bin/sh\n");
      assertTrue(onPath.toFile().setExecutable(true));
      Path sdkBin = Files.createDirectories(home.resolve("bin"));
      Path bundled = sdkBin.resolve(launcher);
      Files.writeString(bundled, "#!/bin/sh\n");
      assertTrue(bundled.toFile().setExecutable(true));

      assertEquals(onPath, KernelChatBench.tornadoExecutable(home.toString(), dir.toString()));
      assertEquals(bundled, KernelChatBench.tornadoExecutable(home.toString(), ""));
      assertNull(KernelChatBench.tornadoExecutable("", dir.resolve("missing").toString()));
    } finally {
      Files.deleteIfExists(dir.resolve(launcher));
      Files.deleteIfExists(home.resolve("bin").resolve(launcher));
      Files.deleteIfExists(home.resolve("bin"));
      Files.deleteIfExists(home);
      Files.deleteIfExists(dir);
    }
  }

  @Test
  void windowsLauncherUsesExeNotExtensionlessScript() throws Exception {
    Path dir = Files.createTempDirectory("tornado-win");
    Path home = Files.createTempDirectory("tornado-win-home");
    try {
      Path script = dir.resolve("tornado");
      Files.writeString(script, "#!/bin/sh\n");
      assertTrue(script.toFile().setExecutable(true));
      Path exe = dir.resolve("tornado.exe");
      Files.writeString(exe, "MZ");
      assertTrue(exe.toFile().setExecutable(true));

      assertEquals(
        exe,
        KernelChatBench.tornadoExecutable(home.toString(), dir.toString(), "Windows 11"));
      assertEquals(List.of("tornado.exe"), KernelChatBench.tornadoLauncherNames("Windows 11"));
      assertEquals(List.of("tornado"), KernelChatBench.tornadoLauncherNames("Linux"));

      Files.delete(exe);
      assertNull(KernelChatBench.tornadoExecutable("", dir.toString(), "Windows 11"));

      Path bundled = Files.createDirectories(home.resolve("bin")).resolve("tornado.exe");
      Files.writeString(bundled, "MZ");
      assertTrue(bundled.toFile().setExecutable(true));
      assertEquals(bundled, KernelChatBench.tornadoExecutable(home.toString(), "", "Windows 11"));
    } finally {
      Files.deleteIfExists(dir.resolve("tornado"));
      Files.deleteIfExists(dir.resolve("tornado.exe"));
      Files.deleteIfExists(home.resolve("bin").resolve("tornado.exe"));
      Files.deleteIfExists(home.resolve("bin"));
      Files.deleteIfExists(home);
      Files.deleteIfExists(dir);
    }
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
    assertTrue(table.contains("skipped: tornado (tornado command not found)"));
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
  void defaultPromptAndBudgetFavorHeavierPrefill() {
    assertTrue(KernelChatBench.CHAT_PROMPT.length() > 80);
    assertEquals(128, KernelChatBench.DEFAULT_MAX_TOKENS);
    assertTrue(BundledModels.isCompactKernelBenchModel(
        Path.of("models/SmolLM2-135M-Instruct-ONNX")));
    assertFalse(BundledModels.isCompactKernelBenchModel(Path.of("models/Qwen3-0.6B")));
  }

  @Test
  void parseOptionsWorkerFlag() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(
      new String[] {"--worker", "--max-tokens", "8", "models/Qwen3-0.6B"});
    assertTrue(options.worker());
    assertEquals(Path.of("models/Qwen3-0.6B"), options.modelPath());
    assertEquals(8, options.maxTokens());
  }

  @Test
  void applicationClasspathPrefersLoaderFiles() throws Exception {
    URL classes = Path.of("/tmp/nano-bench-classes").toUri().toURL();
    URL jar = Path.of("/tmp/nano-vllm-java.jar").toUri().toURL();
    try (URLClassLoader loader = new URLClassLoader(new URL[] {classes, jar}, null)) {
      assertEquals(
        Path.of(classes.toURI()) + File.pathSeparator + Path.of(jar.toURI()),
        KernelChatBench.applicationClasspath(loader, "maven.jar"));
    }
  }

  @Test
  void applicationClasspathFallsBackWhenLoaderHasNoFiles() throws Exception {
    URL remote = URI.create("http://example.invalid/app.jar").toURL();
    try (URLClassLoader loader = new URLClassLoader(new URL[] {remote}, null)) {
      assertEquals("app.jar", KernelChatBench.applicationClasspath(loader, "app.jar"));
    }
    assertEquals("app.jar", KernelChatBench.applicationClasspath(null, "app.jar"));
    assertThrows(
      IllegalStateException.class,
      () -> KernelChatBench.applicationClasspath(null, " "));
  }
}
