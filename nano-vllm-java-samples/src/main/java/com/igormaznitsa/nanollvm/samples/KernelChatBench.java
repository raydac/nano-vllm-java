package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.GenerationStats;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.LlmInImage;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.KernelChatBenchSvg;
import com.igormaznitsa.nanollvm.utils.KernelBackend;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

/**
 * Compares scalar Java, Vector API, and TornadoVM on the same greedy chat. Kernels are
 * process-wide, so this sample starts one child JVM per backend. Scalar and Vector API always run
 * ({@code --add-modules=jdk.incubator.vector} on the child, even when the parent cannot see the
 * incubator module). TornadoVM runs when {@code tornado} ({@code tornado.exe} on Windows) is on
 * {@code PATH} or {@code TORNADOVM_HOME} is set, and that child is started with that launcher. A Maven
 * {@code exec:java} parent cannot see SDK devices, so a parent-only probe would skip TornadoVM.
 * Prints a tok/s table and writes {@code kernel-chat-bench.svg}. When
 * {@code models/YOLO26n/detect/yolo26n.onnx} is present, the same backends then recognize the
 * street photo and write {@code kernel-yolo-bench.svg} in images per second. {@code --skip-yolo}
 * keeps the chat comparison only. {@code --yolo <path>} selects another detection file.
 *
 * <p>From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java
 * -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.KernelChatBench}
 *
 * <p>Args: optional model path; {@code --svg <path>} (default {@code kernel-chat-bench.svg};
 * {@code --svg -} skips the file); {@code --max-tokens N}; {@code --runs N}; {@code --skip-yolo};
 * {@code --yolo <onnx>}. Internal {@code --worker} and {@code --workload chat|yolo} are used by
 * child JVMs. Default checkpoint prefers Qwen3-0.6B / Gemma3-270M over compact ONNX demos so
 * TornadoVM is not judged only on launch overhead.
 */
public final class KernelChatBench {

  static final String RESULT_PREFIX = "KERNEL_RESULT";
  static final String YOLO_RESULT_PREFIX = "YOLO_RESULT";
  static final String DEFAULT_SVG_NAME = "kernel-chat-bench.svg";
  static final String DEFAULT_YOLO_SVG_NAME = "kernel-yolo-bench.svg";
  static final String DEFAULT_BENCH_SVG_NAME = "kernel-bench.svg";
  static final String WORKLOAD_CHAT = "chat";
  static final String WORKLOAD_YOLO = "yolo";
  static final String SVG_PROPERTY = "nanollvm.bench.svg";
  static final String CHAT_PROMPT =
      "Explain why the sky is blue in three short sentences. Mention Rayleigh scattering, "
          + "why shorter wavelengths scatter more, and why sunsets look red near the horizon.";
  static final int DEFAULT_MAX_TOKENS = 128;
  static final int DEFAULT_RUNS = 2;

  private static final List<String> ALL_MODES = List.of("scalar", "vector", "tornado");
  private static final DateTimeFormatter UTC_STAMP =
    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.ROOT).withZone(ZoneOffset.UTC);

  private KernelChatBench() {
  }

  public static void main(final String[] args) throws Exception {
    Options options = Options.parse(args);
    if (options.worker()) {
      runWorker(options);
      return;
    }
    runCoordinator(options);
  }

  static List<String> coordinatorModes(
    final List<String> parentAvailable,
    final boolean tornadoLauncher
  ) {
    requireNonNull(parentAvailable, "parentAvailable");
    Stream.Builder<String> modes = Stream.builder();
    modes.add("scalar");
    modes.add("vector");
    if (tornadoLauncher || parentAvailable.contains("tornado")) {
      modes.add("tornado");
    }
    return modes.build().toList();
  }

  static List<String> skippedModes(final List<String> available) {
    requireNonNull(available, "available");
    return ALL_MODES.stream()
      .filter(mode -> !available.contains(mode))
      .toList();
  }

  static String formatResultLine(final Result result) {
    requireNonNull(result, "result");
    return String.join(
      "\t",
      RESULT_PREFIX,
      result.mode(),
      result.label(),
      Integer.toString(result.promptTokens()),
      Integer.toString(result.completionTokens()),
      Long.toString(result.elapsedNanos()),
      String.format(Locale.ROOT, "%.6f", result.completionTokensPerSecond()));
  }

  static Optional<Result> parseResultLine(final String line) {
    if (line == null || !line.startsWith(RESULT_PREFIX + "\t")) {
      return Optional.empty();
    }
    String[] parts = line.split("\t", -1);
    if (parts.length != 7) {
      return Optional.empty();
    }
    try {
      return Optional.of(new Result(
        parts[1],
        parts[2],
        Integer.parseInt(parts[3]),
        Integer.parseInt(parts[4]),
        Long.parseLong(parts[5]),
        Double.parseDouble(parts[6])));
    } catch (NumberFormatException ignored) {
      return Optional.empty();
    }
  }

  static String formatYoloLine(final YoloResult result) {
    requireNonNull(result, "result");
    return String.join(
      "\t",
      YOLO_RESULT_PREFIX,
      result.mode(),
      result.label(),
      Integer.toString(result.width()),
      Integer.toString(result.height()),
      Long.toString(result.elapsedNanos()),
      String.format(Locale.ROOT, "%.6f", result.imagesPerSecond()));
  }

  static Optional<YoloResult> parseYoloLine(final String line) {
    if (line == null || !line.startsWith(YOLO_RESULT_PREFIX + "\t")) {
      return Optional.empty();
    }
    String[] parts = line.split("\t", -1);
    if (parts.length != 7) {
      return Optional.empty();
    }
    try {
      return Optional.of(new YoloResult(
        parts[1],
        parts[2],
        Integer.parseInt(parts[3]),
        Integer.parseInt(parts[4]),
        Long.parseLong(parts[5]),
        Double.parseDouble(parts[6])));
    } catch (NumberFormatException ignored) {
      return Optional.empty();
    }
  }

  static String formatConsoleTable(
    final Path modelPath,
    final Options options,
    final List<String> skipped,
    final List<Result> results
  ) {
    requireNonNull(modelPath, "modelPath");
    requireNonNull(options, "options");
    requireNonNull(skipped, "skipped");
    requireNonNull(results, "results");
    if (results.isEmpty()) {
      throw new IllegalArgumentException("results must not be empty");
    }

    Result baseline = results.getFirst();
    StringBuilder table = new StringBuilder();
    table.append("Kernel chat benchmark\n");
    table.append("  model: ").append(modelPath).append('\n');
    table.append("  prompt: ").append(CHAT_PROMPT).append('\n');
    table.append("  maxTokens: ").append(options.maxTokens())
      .append("  runs: ").append(options.runs())
      .append(" (plus 1 warmup)\n");
    if (skipped.isEmpty()) {
      table.append("  backends: ")
        .append(results.stream().map(Result::mode).collect(joining(", ")))
        .append('\n');
    } else {
      table.append("  skipped: ").append(describeSkipped(skipped)).append('\n');
    }
    table.append('\n');
    table.append(String.format(
      Locale.ROOT,
      "%-22s %8s %12s %10s %8s %12s%n",
      "Kernel",
      "Prompt",
      "Completion",
      "Seconds",
      "tok/s",
      "vs scalar"));
    for (Result result : results) {
      double speedup = speedupVersus(result, baseline);
      table.append(String.format(
        Locale.ROOT,
        "%-22s %8d %12d %10.2f %8.2f %11.2fx%n",
        result.label(),
        result.promptTokens(),
        result.completionTokens(),
        result.elapsedNanos() / 1e9d,
        result.completionTokensPerSecond(),
        speedup));
    }
    return table.toString();
  }

  static String formatYoloTable(
    final Path modelPath,
    final Options options,
    final List<String> skipped,
    final List<YoloResult> results
  ) {
    requireNonNull(modelPath, "modelPath");
    requireNonNull(options, "options");
    requireNonNull(skipped, "skipped");
    requireNonNull(results, "results");
    if (results.isEmpty()) {
      throw new IllegalArgumentException("results must not be empty");
    }

    YoloResult baseline = results.getFirst();
    StringBuilder table = new StringBuilder();
    table.append("Kernel YOLO benchmark\n");
    table.append("  model: ").append(modelPath).append('\n');
    table.append("  image: ").append(baseline.width()).append('x').append(baseline.height())
      .append(" street photo\n");
    table.append("  runs: ").append(options.runs()).append(" (plus 1 warmup)\n");
    if (skipped.isEmpty()) {
      table.append("  backends: ")
        .append(results.stream().map(YoloResult::mode).collect(joining(", ")))
        .append('\n');
    } else {
      table.append("  skipped: ").append(describeSkipped(skipped)).append('\n');
    }
    table.append('\n');
    table.append(String.format(
      Locale.ROOT,
      "%-22s %10s %10s %8s %12s%n",
      "Kernel",
      "Seconds",
      "ms/image",
      "img/s",
      "vs scalar"));
    for (YoloResult result : results) {
      double seconds = result.elapsedNanos() / 1e9d;
      double millis = options.runs() == 0 ? 0d : seconds / options.runs() * 1e3d;
      table.append(String.format(
        Locale.ROOT,
        "%-22s %10.2f %10.1f %8.2f %11.2fx%n",
        result.label(),
        seconds,
        millis,
        result.imagesPerSecond(),
        yoloSpeedup(result, baseline)));
    }
    return table.toString();
  }

  static String formatSpeedSummary(
    final List<Result> chat,
    final List<YoloResult> yolo,
    final int runs
  ) {
    requireNonNull(chat, "chat");
    requireNonNull(yolo, "yolo");
    if (chat.isEmpty()) {
      throw new IllegalArgumentException("chat results must not be empty");
    }
    if (runs < 1) {
      throw new IllegalArgumentException("runs must be >= 1");
    }

    Result chatBaseline = chat.getFirst();
    YoloResult yoloBaseline = yolo.isEmpty() ? null : yolo.getFirst();
    StringBuilder table = new StringBuilder();
    table.append("Chat and YOLO\n");
    table.append("  timed runs: ").append(runs).append('\n');
    if (yolo.isEmpty()) {
      table.append("  YOLO was not run\n");
    }
    table.append(String.format(
      Locale.ROOT,
      "%-22s %12s %12s %14s %12s %12s%n",
      "Kernel",
      "Chat tok/s",
      "Chat vs",
      "YOLO ms/image",
      "YOLO img/s",
      "YOLO vs"));
    for (Result result : chat) {
      YoloResult picture = yoloFor(yolo, result.mode());
      double chatRate = result.completionTokensPerSecond();
      double chatVersus = chatBaseline.completionTokensPerSecond() <= 0d
        ? 0d
        : chatRate / chatBaseline.completionTokensPerSecond();
      if (picture == null) {
        table.append(String.format(
          Locale.ROOT,
          "%-22s %12.2f %11.2fx %14s %12s %12s%n",
          result.label(),
          chatRate,
          chatVersus,
          "—",
          "—",
          "—"));
      } else {
        double millis = picture.imagesPerSecond() <= 0d
          ? 0d
          : 1000d / picture.imagesPerSecond();
        double yoloVersus = yoloBaseline == null || yoloBaseline.imagesPerSecond() <= 0d
          ? 0d
          : picture.imagesPerSecond() / yoloBaseline.imagesPerSecond();
        table.append(String.format(
          Locale.ROOT,
          "%-22s %12.2f %11.2fx %14.0f %12.2f %11.2fx%n",
          result.label(),
          chatRate,
          chatVersus,
          millis,
          picture.imagesPerSecond(),
          yoloVersus));
      }
    }
    return table.toString();
  }

  private static YoloResult yoloFor(final List<YoloResult> yolo, final String mode) {
    return yolo.stream()
      .filter(result -> result.mode().equals(mode))
      .findFirst()
      .orElse(null);
  }

  private static void runCoordinator(final Options options) throws Exception {
    Path modelPath = resolveModel(options.modelPath());
    boolean tornadoLauncher = tornadoExecutable(
      System.getenv("TORNADOVM_HOME"), System.getenv("PATH")) != null;
    List<String> available = coordinatorModes(KernelBackend.availableModes(), tornadoLauncher);
    List<String> skipped = skippedModes(available);

    System.out.println("Kernel chat benchmark");
    System.out.println("  model: " + modelPath);
    System.out.println("  modes: " + String.join(", ", available));
    if (!skipped.isEmpty()) {
      System.out.println("  skipped: " + describeSkipped(skipped));
    }
    if (BundledModels.isCompactKernelBenchModel(modelPath)) {
      System.out.println(
          "  note: compact checkpoint — TornadoVM tok/s is often launch-bound;"
              + " prefer models/Qwen3-0.6B or models/Gemma3-270M");
    }
    System.out.println();

    List<Result> results = new ArrayList<>();
    for (String mode : available) {
      results.add(runChild(mode, modelPath, options, WORKLOAD_CHAT));
    }

    System.out.println();
    System.out.print(formatConsoleTable(modelPath, options, skipped, results));

    if (options.svgPath() != null) {
      Path svgFile = options.svgPath().toAbsolutePath().normalize();
      Files.writeString(svgFile, renderSvg(modelPath, options, results), UTF_8);
      System.out.println();
      System.out.println("Wrote " + svgFile);
    }
    runYolo(options, available, skipped, modelPath, results);
  }

  private static void runYolo(
    final Options options,
    final List<String> available,
    final List<String> skipped,
    final Path chatModel,
    final List<Result> chat
  ) throws Exception {
    List<YoloResult> yolo = List.of();
    if (!options.skipYolo()) {
      yolo = timeYolo(options, available, skipped);
    }
    System.out.println();
    System.out.print(formatSpeedSummary(chat, yolo, options.runs()));
    Path svgFile = benchSvg(options.svgPath());
    if (svgFile == null) {
      return;
    }
    Files.writeString(
      svgFile.toAbsolutePath().normalize(),
      renderBenchSvg(chatModel, chat, yolo),
      UTF_8);
    System.out.println();
    System.out.println("Wrote " + svgFile.toAbsolutePath().normalize());
  }

  private static List<YoloResult> timeYolo(
    final Options options,
    final List<String> available,
    final List<String> skipped
  ) throws Exception {
    Optional<Path> found = resolveYoloModel(options.yoloPath());
    if (found.isEmpty()) {
      System.out.println();
      System.out.println(
        "YOLO benchmark skipped: models/YOLO26n/detect/yolo26n.onnx is missing.");
      System.out.println(
        "  Run ./models/download-yolo26n.sh to include detection in this comparison.");
      return List.of();
    }

    Path modelPath = found.get();
    System.out.println();
    System.out.println("Kernel YOLO benchmark");
    System.out.println("  model: " + modelPath);
    System.out.println("  modes: " + String.join(", ", available));
    if (!skipped.isEmpty()) {
      System.out.println("  skipped: " + describeSkipped(skipped));
    }
    System.out.println();

    List<YoloResult> results = new ArrayList<>();
    for (String mode : available) {
      results.add(parseYoloLine(runChildLine(mode, modelPath, options, WORKLOAD_YOLO))
        .orElseThrow(() -> new IllegalStateException(
          "no " + YOLO_RESULT_PREFIX + " from worker " + mode)));
    }

    System.out.println();
    System.out.print(formatYoloTable(modelPath, options, skipped, results));
    Path svgFile = yoloSvg(options.svgPath());
    if (svgFile != null) {
      Files.writeString(
        svgFile.toAbsolutePath().normalize(), renderYoloSvg(modelPath, results), UTF_8);
      System.out.println();
      System.out.println("Wrote " + svgFile.toAbsolutePath().normalize());
    }
    return results;
  }

  private static void runWorker(final Options options) throws Exception {
    if (options.yolo()) {
      runYoloWorker(options);
      return;
    }
    Path modelPath = resolveModel(options.modelPath());
    System.out.println("Loading model from " + modelPath);
    System.out.println(KernelBackend.summaryLine());

    SamplingParams sampling = SamplingParams.builder()
      .deterministic()
      .maxTokens(options.maxTokens())
      .ignoreEos(true)
      .build();

    try (LlmModel model = LlmModelFactory.make(modelPath);
         LLM llm = LLM.builder(model).noSystemPrompt().deterministic().build()) {
      if ("tornado".equals(normalizedMode())) {
        System.out.println(
          "TornadoVM compiles each decoder layer on first use. The warmup stays quiet for a few minutes, then generation continues.");
      }
      System.out.println("Warmup generate");
      System.out.flush();
      llm.chat(sampling).send(CHAT_PROMPT);
      System.out.println("Warmup finished");
      System.out.flush();

      long elapsedNanos = 0L;
      int promptTokens = 0;
      int completionTokens = 0;
      for (int run = 0; run < options.runs(); run++) {
        System.out.println("Run " + (run + 1) + " of " + options.runs());
        System.out.flush();
        GenerationStats stats = llm.chat(sampling).send(CHAT_PROMPT).stats();
        elapsedNanos += stats.elapsedNanos();
        promptTokens = stats.promptTokens();
        completionTokens += stats.completionTokens();
      }

      double tokPerSec = elapsedNanos == 0L || completionTokens == 0
        ? 0d
        : completionTokens / (elapsedNanos / 1e9d);
      Result result = new Result(
        normalizedMode(),
        KernelBackend.label(),
        promptTokens,
        completionTokens,
        elapsedNanos,
        tokPerSec);
      System.out.println(formatResultLine(result));
    }
  }

  private static Result runChild(
    final String mode,
    final Path modelPath,
    final Options options,
    final String workload
  ) throws IOException, InterruptedException {
    return parseResultLine(runChildLine(mode, modelPath, options, workload)).orElseThrow(
      () -> new IllegalStateException("no " + RESULT_PREFIX + " from worker " + mode));
  }

  private static String runChildLine(
    final String mode,
    final Path modelPath,
    final Options options,
    final String workload
  ) throws IOException, InterruptedException {
    List<String> command = childCommand(mode, modelPath, options, workload);
    System.out.println(">> " + mode + " (" + String.join(" ", command.subList(0, 1))
      + " -D" + PROP_KERNELS + "=" + mode + " " + workload + ")");

    ProcessBuilder builder = new ProcessBuilder(command);
    if ("tornado".equals(mode)) {
      publishTornadoHome(builder, command);
      // TornadoVM's launcher appends $CLASSPATH with a Unix ':' even on Windows.
      builder.environment().remove("CLASSPATH");
    }
    builder.redirectError(ProcessBuilder.Redirect.INHERIT);
    Process process = builder.start();

    String captured = null;
    try (BufferedReader reader = process.inputReader(UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (isResultLine(line)) {
          captured = line;
        } else {
          System.out.println(line);
        }
      }
    }

    int exit = process.waitFor();
    if (exit != 0) {
      throw new IllegalStateException(
        "kernel worker %s exited %d".formatted(mode, exit));
    }
    if (captured == null) {
      throw new IllegalStateException("no result line from worker " + mode);
    }
    return captured;
  }

  private static boolean isResultLine(final String line) {
    return line.startsWith(RESULT_PREFIX + "\t") || line.startsWith(YOLO_RESULT_PREFIX + "\t");
  }

  private static List<String> childCommand(
    final String mode,
    final Path modelPath,
    final Options options,
    final String workload
  ) {
    String classpath = applicationClasspath(
      Thread.currentThread().getContextClassLoader(),
      System.getProperty("java.class.path"));
    if ("tornado".equals(mode)) {
      Path tornado = tornadoExecutable(System.getenv("TORNADOVM_HOME"), System.getenv("PATH"));
      if (tornado != null) {
        return tornadoChildCommand(
          tornado, classpath, modelPath.toString(), options.maxTokens(), options.runs(), workload);
      }
    }

    List<String> command = new ArrayList<>();
    command.add(javaExecutable());
    command.addAll(inheritedJvmArgs(mode));
    command.add("-cp");
    command.add(classpath);
    command.add(KernelChatBench.class.getName());
    command.add("--worker");
    command.add("--workload");
    command.add(workload);
    command.add("--max-tokens");
    command.add(Integer.toString(options.maxTokens()));
    command.add("--runs");
    command.add(Integer.toString(options.runs()));
    command.add(modelPath.toString());
    return List.copyOf(command);
  }

  static List<String> tornadoChildCommand(
    final Path tornado,
    final String classpath,
    final String modelPath,
    final int maxTokens,
    final int runs
  ) {
    return tornadoChildCommand(tornado, classpath, modelPath, maxTokens, runs, WORKLOAD_CHAT);
  }

  static List<String> tornadoChildCommand(
    final Path tornado,
    final String classpath,
    final String modelPath,
    final int maxTokens,
    final int runs,
    final String workload
  ) {
    requireNonNull(tornado, "tornado");
    requireNonNull(classpath, "classpath");
    requireNonNull(modelPath, "modelPath");
    requireNonNull(workload, "workload");
    return List.of(
      tornado.toString(),
      "--classpath",
        // TornadoVM pastes --classpath into os.system(); cmd.exe splits on unquoted ';'.
        shellQuote(classpath),
      "--jvm=-D" + PROP_KERNELS + "=tornado -Dtornado.enabled.backends=OpenCL"
        + " --add-modules=jdk.incubator.vector",
      "--params=" + workerArguments(modelPath, maxTokens, runs, workload),
      KernelChatBench.class.getName());
  }

  static String workerArguments(final String modelPath, final int maxTokens, final int runs) {
    return workerArguments(modelPath, maxTokens, runs, WORKLOAD_CHAT);
  }

  static String workerArguments(
    final String modelPath, final int maxTokens, final int runs, final String workload
  ) {
    requireNonNull(modelPath, "modelPath");
    requireNonNull(workload, "workload");
    if (!WORKLOAD_CHAT.equals(workload) && !WORKLOAD_YOLO.equals(workload)) {
      throw new IllegalArgumentException("workload must be chat or yolo");
    }
    return "--worker --workload %s --max-tokens %d --runs %d %s".formatted(
      workload, maxTokens, runs, shellQuote(modelPath));
  }

  static String shellQuote(final String value) {
    return shellQuote(value, System.getProperty("os.name", ""));
  }

  static String shellQuote(final String value, final String osName) {
    requireNonNull(value, "value");
    requireNonNull(osName, "osName");
    if (osName.toLowerCase(Locale.ROOT).contains("win")) {
      return "\"" + value.replace("\"", "\"\"") + "\"";
    }
    return "'" + value.replace("'", "'\\''") + "'";
  }

  static Path tornadoExecutable(final String tornadoHome, final String pathEnv) {
    return tornadoExecutable(tornadoHome, pathEnv, System.getProperty("os.name", ""));
  }

  static Path tornadoExecutable(
    final String tornadoHome,
    final String pathEnv,
    final String osName
  ) {
    requireNonNull(osName, "osName");
    List<String> names = tornadoLauncherNames(osName);
    Path onPath = firstExecutable(pathEnv, names);
    if (onPath != null) {
      return onPath;
    }
    if (tornadoHome == null || tornadoHome.isBlank()) {
      return null;
    }
    return names.stream()
      .map(name -> Path.of(tornadoHome, "bin", name))
      .filter(Files::isExecutable)
      .findFirst()
      .orElse(null);
  }

  static List<String> tornadoLauncherNames(final String osName) {
    requireNonNull(osName, "osName");
    return osName.toLowerCase(Locale.ROOT).contains("win")
      ? List.of("tornado.exe")
      : List.of("tornado");
  }

  static Path sdkHome(final Path executable) {
    requireNonNull(executable, "executable");
    Path bin = executable.toAbsolutePath().normalize().getParent();
    if (bin == null || !"bin".equals(String.valueOf(bin.getFileName()))) {
      return null;
    }
    Path sdk = bin.getParent();
    if (sdk != null && Files.isDirectory(sdk.resolve("share").resolve("java").resolve("tornado"))) {
      return sdk;
    }
    return null;
  }

  private static Path firstExecutable(final String pathEnv, final List<String> names) {
    if (pathEnv == null || pathEnv.isBlank()) {
      return null;
    }
    return Arrays.stream(pathEnv.split(File.pathSeparator))
      .filter(dir -> !dir.isBlank())
      .flatMap(dir -> names.stream().map(name -> Path.of(dir, name)))
      .filter(Files::isExecutable)
      .findFirst()
      .orElse(null);
  }

  private static void publishTornadoHome(final ProcessBuilder builder, final List<String> command) {
    if (!builder.environment().getOrDefault("TORNADOVM_HOME", "").isBlank() || command.isEmpty()) {
      return;
    }
    Path sdk = sdkHome(Path.of(command.getFirst()));
    if (sdk != null) {
      builder.environment().put("TORNADOVM_HOME", sdk.toString());
    }
  }

  static String applicationClasspath(final ClassLoader loader, final String javaClassPath) {
    String fromLoader = classpathOf(loader);
    if (!fromLoader.isBlank()) {
      return fromLoader;
    }
    if (javaClassPath == null || javaClassPath.isBlank()) {
      throw new IllegalStateException("java.class.path is empty");
    }
    return javaClassPath;
  }

  private static String classpathOf(final ClassLoader loader) {
    if (!(loader instanceof URLClassLoader urls) || urls.getURLs().length == 0) {
      return "";
    }
    return Arrays.stream(urls.getURLs())
      .filter(url -> "file".equals(url.getProtocol()))
      .map(KernelChatBench::filePath)
      .collect(joining(File.pathSeparator));
  }

  private static String filePath(final URL url) {
    try {
      return Path.of(url.toURI()).toString();
    } catch (URISyntaxException failed) {
      throw new IllegalStateException("classpath URL is not a file: " + url, failed);
    }
  }

  private static List<String> inheritedJvmArgs(final String mode) {
    List<String> args = new ArrayList<>();
    for (String arg : ManagementFactory.getRuntimeMXBean().getInputArguments()) {
      if (isInheritedJvmArg(arg)) {
        args.add(arg);
      }
    }
    args.add("--add-modules=jdk.incubator.vector");
    args.add("-D" + PROP_KERNELS + "=" + mode);
    return args;
  }

  private static boolean isInheritedJvmArg(final String arg) {
    if (arg.startsWith("-D" + PROP_KERNELS + "=") || arg.startsWith("-D" + SVG_PROPERTY + "=")) {
      return false;
    }
    return !arg.startsWith("-agent") && !arg.startsWith("-javaagent:");
  }

  private static String javaExecutable() {
    Path bundled = Path.of(System.getProperty("java.home"), "bin", "java");
    return Files.isExecutable(bundled)
      ? bundled.toString()
      : ProcessHandle.current().info().command().orElse("java");
  }

  private static Path resolveModel(final Path cliPath) {
    if (cliPath != null) {
      return cliPath.toAbsolutePath().normalize();
    }
    return BundledModels.requireKernelChatBench();
  }

  private static String normalizedMode() {
    String mode = KernelBackend.mode();
    return switch (mode) {
      case "plain" -> "scalar";
      case "simd" -> "vector";
      case "gpu" -> "tornado";
      default -> mode;
    };
  }

  private static String describeSkipped(final List<String> skipped) {
    return skipped.stream()
      .map(mode -> mode + " (" + skipReason(mode) + ")")
      .collect(joining("; "));
  }

  private static String skipReason(final String mode) {
    return switch (mode) {
      case "vector" -> "Vector API unavailable";
      case "tornado" -> "tornado command not found";
      default -> "unavailable";
    };
  }

  private static double speedupVersus(final Result result, final Result baseline) {
    if (baseline.completionTokensPerSecond() <= 0d) {
      return 0d;
    }
    return result.completionTokensPerSecond() / baseline.completionTokensPerSecond();
  }

  private static double yoloSpeedup(final YoloResult result, final YoloResult baseline) {
    if (baseline.imagesPerSecond() <= 0d) {
      return 0d;
    }
    return result.imagesPerSecond() / baseline.imagesPerSecond();
  }

  private static void runYoloWorker(final Options options) throws Exception {
    Path modelPath = options.modelPath();
    if (modelPath == null || !Files.isRegularFile(modelPath)) {
      throw new IllegalStateException("YOLO weights not found: " + modelPath);
    }
    ArgbImage image = benchImage();
    System.out.println("Loading model from " + modelPath);
    System.out.println(KernelBackend.summaryLine());
    System.out.println("Image " + image.width() + "x" + image.height());

    LlmInImage input = new LlmInImage(image);
    try (LlmModel model = LlmModelFactory.make(modelPath);
         LLM llm = LLM.builder(model).allCpuThreads().build()) {
      if ("tornado".equals(normalizedMode())) {
        System.out.println(
          "TornadoVM compiles each convolution on first use. The warmup stays quiet, then recognition continues.");
      }
      System.out.println("Warmup recognize");
      System.out.flush();
      llm.generate(input, LlmModality.IMAGE);
      System.out.println("Warmup finished");
      System.out.flush();

      long elapsedNanos = 0L;
      for (int run = 0; run < options.runs(); run++) {
        System.out.println("Run " + (run + 1) + " of " + options.runs());
        System.out.flush();
        long started = System.nanoTime();
        llm.generate(input, LlmModality.IMAGE);
        elapsedNanos += System.nanoTime() - started;
      }

      double imagesPerSecond = elapsedNanos == 0L
        ? 0d
        : options.runs() / (elapsedNanos / 1e9d);
      System.out.println(formatYoloLine(new YoloResult(
        normalizedMode(),
        KernelBackend.label(),
        image.width(),
        image.height(),
        elapsedNanos,
        imagesPerSecond)));
    }
  }

  static Optional<Path> resolveYoloModel(final Path cliPath) {
    if (cliPath != null) {
      Path path = cliPath.toAbsolutePath().normalize();
      if (!Files.isRegularFile(path)) {
        throw new IllegalStateException("YOLO weights not found: " + path);
      }
      return Optional.of(path);
    }
    Path underRoot = BundledModels.modelsRoot().resolve("YOLO26n").resolve("detect")
      .resolve("yolo26n.onnx");
    if (Files.isRegularFile(underRoot)) {
      return Optional.of(underRoot.toAbsolutePath().normalize());
    }
    Path cwd = Path.of("models", "YOLO26n", "detect", "yolo26n.onnx").toAbsolutePath().normalize();
    if (Files.isRegularFile(cwd)) {
      return Optional.of(cwd);
    }
    return Optional.empty();
  }

  static Path yoloSvg(final Path chatSvg) {
    return siblingSvg(chatSvg, DEFAULT_YOLO_SVG_NAME);
  }

  static Path benchSvg(final Path chatSvg) {
    return siblingSvg(chatSvg, DEFAULT_BENCH_SVG_NAME);
  }

  private static Path siblingSvg(final Path chatSvg, final String name) {
    if (chatSvg == null) {
      return null;
    }
    Path parent = chatSvg.getParent();
    return parent == null ? Path.of(name) : parent.resolve(name);
  }

  private static ArgbImage benchImage() throws IOException {
    try (InputStream source = KernelChatBench.class.getResourceAsStream(
      YoloStreetHelloWorld.STREET_PHOTO)) {
      if (source != null) {
        BufferedImage photo = ImageIO.read(source);
        if (photo != null) {
          return YoloStreetHelloWorld.fromBufferedImage(photo);
        }
      }
    }
    int width = 800;
    int height = 544;
    int[] pixels = new int[width * height];
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = 0xff000000 | ((i * 13) & 0xffffff);
    }
    return new ArgbImage(width, height, pixels);
  }

  private static String renderYoloSvg(final Path modelPath, final List<YoloResult> results) {
    return KernelChatBenchSvg.render(
      "YOLO speed",
      yoloSubtitle(modelPath, results.getFirst()),
      "img/s",
      yoloBars(results));
  }

  private static String renderBenchSvg(
    final Path chatModel, final List<Result> chat, final List<YoloResult> yolo
  ) {
    Result baseline = chat.getFirst();
    List<KernelChatBenchSvg.Bar> chatBars = chat.stream()
      .map(result -> new KernelChatBenchSvg.Bar(
        result.mode(),
        result.label(),
        result.completionTokensPerSecond(),
        speedupVersus(result, baseline)))
      .toList();
    List<KernelChatBenchSvg.Section> sections = new ArrayList<>();
    sections.add(new KernelChatBenchSvg.Section("Chat · tokens per second", "tok/s", chatBars));
    if (!yolo.isEmpty()) {
      sections.add(new KernelChatBenchSvg.Section(
        "YOLO · time per street photo", "img/s", yoloBars(yolo)));
    }
    Path fileName = chatModel.getFileName();
    String modelName = fileName == null ? chatModel.toString() : fileName.toString();
    String subtitle = yolo.isEmpty()
      ? "%s · %s".formatted(modelName, UTC_STAMP.format(Instant.now()))
      : "%s · YOLO %s · %s".formatted(
      modelName, yoloSubtitle(null, yolo.getFirst()), UTC_STAMP.format(Instant.now()));
    return KernelChatBenchSvg.renderSections("Chat and YOLO", subtitle, sections);
  }

  private static List<KernelChatBenchSvg.Bar> yoloBars(final List<YoloResult> results) {
    YoloResult baseline = results.getFirst();
    return results.stream()
      .map(result -> {
        double speedup = yoloSpeedup(result, baseline);
        double millis = result.imagesPerSecond() <= 0d
          ? 0d
          : 1000d / result.imagesPerSecond();
        String caption = String.format(
          Locale.ROOT, "%.0f ms · %.2f img/s (%.2fx)", millis, result.imagesPerSecond(), speedup);
        return new KernelChatBenchSvg.Bar(
          result.mode(), result.label(), result.imagesPerSecond(), speedup, caption);
      })
      .toList();
  }

  private static String yoloSubtitle(final Path modelPath, final YoloResult picture) {
    String size = "%dx%d".formatted(picture.width(), picture.height());
    if (modelPath == null) {
      return size;
    }
    Path fileName = modelPath.getFileName();
    String modelName = fileName == null ? modelPath.toString() : fileName.toString();
    return "%s · %s · %s".formatted(modelName, size, UTC_STAMP.format(Instant.now()));
  }

  private static String renderSvg(
    final Path modelPath,
    final Options options,
    final List<Result> results
  ) {
    Result baseline = results.getFirst();
    List<KernelChatBenchSvg.Bar> bars = results.stream()
      .map(result -> new KernelChatBenchSvg.Bar(
        result.mode(),
        result.label(),
        result.completionTokensPerSecond(),
        speedupVersus(result, baseline)))
      .toList();
    Path fileName = modelPath.getFileName();
    String modelName = fileName == null ? modelPath.toString() : fileName.toString();
    String subtitle = "%s · maxTokens=%d · %s".formatted(
      modelName,
      options.maxTokens(),
      UTC_STAMP.format(Instant.now()));
    return KernelChatBenchSvg.render("Kernel chat throughput", subtitle, bars);
  }

  record Result(
    String mode,
    String label,
    int promptTokens,
    int completionTokens,
    long elapsedNanos,
    double completionTokensPerSecond
  ) {
    Result {
      requireNonNull(mode, "mode");
      requireNonNull(label, "label");
    }
  }

  record YoloResult(
    String mode,
    String label,
    int width,
    int height,
    long elapsedNanos,
    double imagesPerSecond
  ) {
    YoloResult {
      requireNonNull(mode, "mode");
      requireNonNull(label, "label");
    }
  }

  record Options(
    boolean worker,
    String workload,
    Path modelPath,
    Path yoloPath,
    boolean skipYolo,
    Path svgPath,
    int maxTokens,
    int runs
  ) {
    static Options parse(final String[] args) {
      boolean worker = false;
      String workload = WORKLOAD_CHAT;
      Path modelPath = null;
      Path yoloPath = null;
      boolean skipYolo = false;
      Path svgPath = Path.of(DEFAULT_SVG_NAME);
      boolean svgFromCli = false;
      int maxTokens = DEFAULT_MAX_TOKENS;
      int runs = DEFAULT_RUNS;

      String[] argv = args == null ? new String[0] : args;
      for (int index = 0; index < argv.length; index++) {
        String arg = argv[index];
        if ("--worker".equals(arg)) {
          worker = true;
        } else if ("--workload".equals(arg)) {
          workload = requireValue(argv, index, "--workload");
          index++;
        } else if ("--skip-yolo".equals(arg)) {
          skipYolo = true;
        } else if ("--yolo".equals(arg)) {
          yoloPath = Path.of(requireValue(argv, index, "--yolo"));
          index++;
        } else if ("--svg".equals(arg)) {
          String value = requireValue(argv, index, "--svg");
          index++;
          svgFromCli = true;
          svgPath = "-".equals(value) ? null : Path.of(value);
        } else if ("--max-tokens".equals(arg)) {
          maxTokens = Integer.parseInt(requireValue(argv, index, "--max-tokens"));
          index++;
        } else if ("--runs".equals(arg)) {
          runs = Integer.parseInt(requireValue(argv, index, "--runs"));
          index++;
        } else if (arg.startsWith("-")) {
          throw new IllegalArgumentException("unknown argument: " + arg);
        } else if (modelPath == null) {
          modelPath = Path.of(arg);
        } else {
          throw new IllegalArgumentException("unexpected extra argument: " + arg);
        }
      }

      if (!WORKLOAD_CHAT.equals(workload) && !WORKLOAD_YOLO.equals(workload)) {
        throw new IllegalArgumentException("workload must be chat or yolo");
      }
      if (!worker && !svgFromCli) {
        svgPath = svgFromPropertyOrDefault(svgPath);
      }
      if (maxTokens < 1) {
        throw new IllegalArgumentException("maxTokens must be >= 1");
      }
      if (runs < 1) {
        throw new IllegalArgumentException("runs must be >= 1");
      }
      return new Options(
        worker, workload, modelPath, yoloPath, skipYolo, svgPath, maxTokens, runs);
    }

    boolean yolo() {
      return WORKLOAD_YOLO.equals(this.workload);
    }

    private static String requireValue(final String[] argv, final int index, final String flag) {
      if (index + 1 >= argv.length) {
        throw new IllegalArgumentException(flag + " requires a value");
      }
      return argv[index + 1];
    }

    private static Path svgFromPropertyOrDefault(final Path defaultPath) {
      String property = System.getProperty(SVG_PROPERTY);
      if (property == null || property.isBlank()) {
        return defaultPath;
      }
      String value = property.strip();
      return "-".equals(value) ? null : Path.of(value);
    }
  }
}
