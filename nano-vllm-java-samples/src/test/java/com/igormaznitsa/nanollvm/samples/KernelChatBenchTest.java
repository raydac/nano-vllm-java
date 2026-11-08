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
    assertTrue(command.stream().anyMatch(arg ->
      arg.contains("--workload " + KernelChatBench.WORKLOAD_CHAT)));
  }

  @Test
  void tornadoYoloChildNamesThePictureWorkload() {
    List<String> command = KernelChatBench.tornadoChildCommand(
      Path.of("/opt/tornado"),
      "classes",
      "/models/yolo26n.onnx",
      8,
      2,
      KernelChatBench.WORKLOAD_YOLO);
    assertTrue(command.stream().anyMatch(arg ->
      arg.contains("--workload yolo")
        && arg.contains(KernelChatBench.shellQuote("/models/yolo26n.onnx"))));
  }

  @Test
  void yoloResultLineRoundTrips() {
    KernelChatBench.YoloResult original = new KernelChatBench.YoloResult(
      "vector", "Vector API (SIMD)", 800, 544, 940_000_000L, 2.12766d);
    String line = KernelChatBench.formatYoloLine(original);
    KernelChatBench.YoloResult result = KernelChatBench.parseYoloLine(line).orElseThrow();
    assertEquals(original.mode(), result.mode());
    assertEquals(original.label(), result.label());
    assertEquals(original.width(), result.width());
    assertEquals(original.height(), result.height());
    assertEquals(original.elapsedNanos(), result.elapsedNanos());
    assertEquals(original.imagesPerSecond(), result.imagesPerSecond(), 1e-6);
    assertTrue(KernelChatBench.parseYoloLine("KERNEL_RESULT\tx").isEmpty());
  }

  @Test
  void yoloTableReportsMillisecondsAndImagesPerSecond() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(
      new String[] {"--svg", "-", "--runs", "2", "--skip-yolo"});
    List<KernelChatBench.YoloResult> results = List.of(
      new KernelChatBench.YoloResult("scalar", "Scalar Java", 800, 544, 2_000_000_000L, 1.0d),
      new KernelChatBench.YoloResult("vector", "Vector API (SIMD)", 800, 544, 1_000_000_000L,
        2.0d));
    String table = KernelChatBench.formatYoloTable(
      Path.of("models/YOLO26n/detect/yolo26n.onnx"),
      options,
      List.of(),
      results);

    assertTrue(table.contains("800x544"));
    assertTrue(table.contains("1000.0"));
    assertTrue(table.contains("500.0"));
    assertTrue(table.contains("2.00x"));
    assertTrue(options.skipYolo());
    assertEquals(
      Path.of("out", "kernel-yolo-bench.svg"),
      KernelChatBench.yoloSvg(Path.of("out", "kernel-chat-bench.svg")));
    assertNull(KernelChatBench.yoloSvg(null));
    assertEquals(
      Path.of("out", "kernel-bench.svg"),
      KernelChatBench.benchSvg(Path.of("out", "kernel-chat-bench.svg")));
  }

  @Test
  void speedSummaryShowsChatAndYoloOnSeparateColumns() {
    List<KernelChatBench.Result> chat = List.of(
      new KernelChatBench.Result("scalar", "Scalar Java", 18, 64, 10_000_000_000L, 6.4d),
      new KernelChatBench.Result("vector", "Vector API (SIMD)", 18, 64, 4_000_000_000L, 16.0d));
    List<KernelChatBench.YoloResult> yolo = List.of(
      new KernelChatBench.YoloResult("scalar", "Scalar Java", 800, 544, 2_000_000_000L, 1.0d),
      new KernelChatBench.YoloResult("vector", "Vector API (SIMD)", 800, 544, 1_000_000_000L,
        2.0d));
    String table = KernelChatBench.formatSpeedSummary(chat, yolo, 2);

    assertTrue(table.contains("Chat tok/s"));
    assertTrue(table.contains("YOLO ms/image"));
    assertTrue(table.contains("6.40"));
    assertTrue(table.contains("16.00"));
    assertTrue(table.contains("1000"));
    assertTrue(table.contains("500"));
    assertTrue(table.contains("2.50x"));
    assertTrue(table.contains("2.00x"));
  }

  @Test
  void speechResultLineRoundTrips() {
    KernelChatBench.SpeechResult original = new KernelChatBench.SpeechResult(
      "vector", "Vector API (SIMD)", 1_500_000_000L, 7_000_000_000L, 1.333333d, 0.285714d);
    String line = KernelChatBench.formatSpeechLine(original);
    KernelChatBench.SpeechResult result = KernelChatBench.parseSpeechLine(line).orElseThrow();
    assertEquals(original.mode(), result.mode());
    assertEquals(original.label(), result.label());
    assertEquals(original.synthNanos(), result.synthNanos());
    assertEquals(original.decodeNanos(), result.decodeNanos());
    assertEquals(original.synthPerSecond(), result.synthPerSecond(), 1e-6d);
    assertEquals(original.decodePerSecond(), result.decodePerSecond(), 1e-6d);
  }

  @Test
  void speedSummaryShowsSpeechSecondsBesideChat() {
    List<KernelChatBench.Result> chat = List.of(
      new KernelChatBench.Result("scalar", "Scalar Java", 18, 64, 10_000_000_000L, 6.4d));
    List<KernelChatBench.SpeechResult> speech = List.of(
      new KernelChatBench.SpeechResult("scalar", "Scalar Java", 2_000_000_000L, 8_000_000_000L,
        1.0d, 0.25d),
      new KernelChatBench.SpeechResult("vector", "Vector API (SIMD)", 1_000_000_000L,
        4_000_000_000L, 2.0d, 0.5d));
    String table = KernelChatBench.formatSpeedSummary(chat, List.of(), speech, 2);

    assertTrue(table.contains("Speech · Hello, World."));
    assertTrue(table.contains("Synth s"));
    assertTrue(table.contains("Decode s"));
    assertTrue(table.contains("1.00"));
    assertTrue(table.contains("4.00"));
    assertTrue(table.contains("2.00x"));
    assertEquals(
      Path.of("out", "kernel-speech-bench.svg"),
      KernelChatBench.speechSvg(Path.of("out", "kernel-chat-bench.svg")));
    assertNull(KernelChatBench.speechSvg(null));
  }

  @Test
  void tornadoSpeechChildNamesTheVoiceAndTheCheckpoint() {
    List<String> command = KernelChatBench.tornadoChildCommand(
      Path.of("/opt/tornado"),
      "classes",
      "/models/whisper-tiny",
      8,
      2,
      KernelChatBench.WORKLOAD_SPEECH,
      "/models/whisper-tiny",
      "/models/piper-en-lessac-medium");
    assertTrue(command.stream().anyMatch(arg ->
      arg.contains("--workload speech")
        && arg.contains("--whisper")
        && arg.contains("--piper")
        && arg.contains(KernelChatBench.shellQuote("/models/whisper-tiny"))
        && arg.contains(KernelChatBench.shellQuote("/models/piper-en-lessac-medium"))));
  }

  @Test
  void parseOptionsReadsSpeechFolders() {
    KernelChatBench.Options options = KernelChatBench.Options.parse(new String[] {
      "--skip-speech",
      "--whisper", "models/whisper-base",
      "--piper", "models/piper-en-lessac-medium"
    });
    assertTrue(options.skipSpeech());
    assertEquals(Path.of("models/whisper-base"), options.whisperPath());
    assertEquals(Path.of("models/piper-en-lessac-medium"), options.piperPath());
    KernelChatBench.Options bound = options.withSpeech(Path.of("w"), Path.of("p"));
    assertFalse(bound.speech());
    assertEquals(Path.of("w"), bound.whisperPath());
    assertEquals(Path.of("p"), bound.piperPath());
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
