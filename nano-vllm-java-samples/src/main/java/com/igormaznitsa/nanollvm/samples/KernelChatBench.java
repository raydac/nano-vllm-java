package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_KERNELS;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.GenerationStats;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.KernelChatBenchSvg;
import com.igormaznitsa.nanollvm.utils.KernelBackend;
import java.io.BufferedReader;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Compares scalar Java, Vector API, and TornadoVM (when a device is present) on the same greedy
 * chat. Kernels are process-wide, so this sample starts one child JVM per backend. Scalar and
 * Vector API always run ({@code --add-modules=jdk.incubator.vector} on the child, even when the
 * parent is the TornadoVM launcher and cannot see the incubator module). TornadoVM runs when the
 * parent reports a device. Prints a tok/s table and writes {@code kernel-chat-bench.svg}.
 *
 * <p>From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java
 * -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.KernelChatBench}
 *
 * <p>Args: optional model path; {@code --svg <path>} (default {@code kernel-chat-bench.svg};
 * {@code --svg -} skips the file); {@code --max-tokens N}; {@code --runs N}. Internal
 * {@code --worker} is used by child JVMs.
 */
public final class KernelChatBench {

  static final String RESULT_PREFIX = "KERNEL_RESULT";
  static final String DEFAULT_SVG_NAME = "kernel-chat-bench.svg";
  static final String SVG_PROPERTY = "nanollvm.bench.svg";
  static final String CHAT_PROMPT = "Explain why the sky is blue in three short sentences.";
  static final int DEFAULT_MAX_TOKENS = 64;
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

  static List<String> coordinatorModes(final List<String> parentAvailable) {
    requireNonNull(parentAvailable, "parentAvailable");
    Stream.Builder<String> modes = Stream.builder();
    modes.add("scalar");
    modes.add("vector");
    if (parentAvailable.contains("tornado")) {
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

  private static void runCoordinator(final Options options) throws Exception {
    Path modelPath = resolveModel(options.modelPath());
    List<String> available = coordinatorModes(KernelBackend.availableModes());
    List<String> skipped = skippedModes(available);

    System.out.println("Kernel chat benchmark");
    System.out.println("  model: " + modelPath);
    System.out.println("  modes: " + String.join(", ", available));
    if (!skipped.isEmpty()) {
      System.out.println("  skipped: " + describeSkipped(skipped));
    }
    System.out.println();

    List<Result> results = new ArrayList<>();
    for (String mode : available) {
      results.add(runChild(mode, modelPath, options));
    }

    System.out.println();
    System.out.print(formatConsoleTable(modelPath, options, skipped, results));

    if (options.svgPath() == null) {
      return;
    }
    Path svgFile = options.svgPath().toAbsolutePath().normalize();
    Files.writeString(svgFile, renderSvg(modelPath, options, results), UTF_8);
    System.out.println();
    System.out.println("Wrote " + svgFile);
  }

  private static void runWorker(final Options options) {
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
      llm.chat(sampling).send(CHAT_PROMPT);

      long elapsedNanos = 0L;
      int promptTokens = 0;
      int completionTokens = 0;
      for (int run = 0; run < options.runs(); run++) {
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
    final Options options
  ) throws IOException, InterruptedException {
    List<String> command = childCommand(mode, modelPath, options);
    System.out.println(">> " + mode + " (" + String.join(" ", command.subList(0, 1))
      + " -D" + PROP_KERNELS + "=" + mode + ")");

    ProcessBuilder builder = new ProcessBuilder(command);
    builder.redirectError(ProcessBuilder.Redirect.INHERIT);
    Process process = builder.start();

    Optional<Result> parsed = Optional.empty();
    try (BufferedReader reader = process.inputReader(UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        Optional<Result> result = parseResultLine(line);
        if (result.isPresent()) {
          parsed = result;
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
    return parsed.orElseThrow(
      () -> new IllegalStateException("no " + RESULT_PREFIX + " from worker " + mode));
  }

  private static List<String> childCommand(
    final String mode,
    final Path modelPath,
    final Options options
  ) {
    String classpath = System.getProperty("java.class.path");
    if (classpath == null || classpath.isBlank()) {
      throw new IllegalStateException("java.class.path is empty");
    }

    List<String> command = new ArrayList<>();
    command.add(javaExecutable());
    command.addAll(inheritedJvmArgs(mode));
    command.add("-cp");
    command.add(classpath);
    command.add(KernelChatBench.class.getName());
    command.add("--worker");
    command.add("--max-tokens");
    command.add(Integer.toString(options.maxTokens()));
    command.add("--runs");
    command.add(Integer.toString(options.runs()));
    command.add(modelPath.toString());
    return List.copyOf(command);
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
    return BundledModels.requireChatDemo();
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
      case "tornado" -> "TornadoVM unavailable (no SDK/device)";
      default -> "unavailable";
    };
  }

  private static double speedupVersus(final Result result, final Result baseline) {
    if (baseline.completionTokensPerSecond() <= 0d) {
      return 0d;
    }
    return result.completionTokensPerSecond() / baseline.completionTokensPerSecond();
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

  record Options(
    boolean worker,
    Path modelPath,
    Path svgPath,
    int maxTokens,
    int runs
  ) {
    static Options parse(final String[] args) {
      boolean worker = false;
      Path modelPath = null;
      Path svgPath = Path.of(DEFAULT_SVG_NAME);
      boolean svgFromCli = false;
      int maxTokens = DEFAULT_MAX_TOKENS;
      int runs = DEFAULT_RUNS;

      String[] argv = args == null ? new String[0] : args;
      for (int index = 0; index < argv.length; index++) {
        String arg = argv[index];
        if ("--worker".equals(arg)) {
          worker = true;
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

      if (!worker && !svgFromCli) {
        svgPath = svgFromPropertyOrDefault(svgPath);
      }
      if (maxTokens < 1) {
        throw new IllegalArgumentException("maxTokens must be >= 1");
      }
      if (runs < 1) {
        throw new IllegalArgumentException("runs must be >= 1");
      }
      return new Options(worker, modelPath, svgPath, maxTokens, runs);
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
