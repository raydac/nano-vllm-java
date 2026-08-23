package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.rag.DenseRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.LabeledText;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.Prediction;
import com.igormaznitsa.nanollvm.samples.utils.OrderedConsole;
import com.igormaznitsa.nanollvm.samples.utils.SampleAdvisorPrompts;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Interactive terminal sample: pick a model, pick a RAG mode, pick how many advisors, then chat
 * (or embed / few-shot classify).
 *
 * <p>Read {@link #main} top to bottom. Each RAG / embed path is a named method that shows the
 * matching library API in order.
 *
 * <p>Launch from the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java}
 *
 * <p>Pass {@code --debug} to print prepared model-user prompts ({@code debug>} on stderr).
 * Off by default.
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;
  private static final int COMPACT_DEMO_MAX_NEW_TOKENS = 256;
  private static final int RAG_MAX_TOKENS_DEFAULT = 768;
  private static final int RAG_MAX_TOKENS_TURN_BASED = 128;
  private static final int RAG_TOP_K_DEFAULT = 4;
  private static final int RAG_TOP_K_TURN_BASED = 2;
  private static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  private static final int RAG_CONTEXT_CHARS_TURN_BASED = 900;
  private static final int EMBED_PREVIEW = 8;
  private static final double CLASSIFY_CLOSE_MARGIN = 0.03;
  private static final String CLASSIFY_LABELS_EXAMPLE =
    "nano-vllm-java-samples/classify-labels.example.txt";
  private static final String DEBUG_FLAG = "--debug";
  private static final List<LabeledText> CLASSIFY_DEMO = List.of(
    new LabeledText("poem", "Однажды в студёную зимнюю пору"),
    new LabeledText("poem", "Я из лесу вышел — был сильный мороз"),
    new LabeledText("poem", "Мороз и солнце; день чудесный"),
    new LabeledText("chat", "who are you?"),
    new LabeledText("chat", "Oh my God"),
    new LabeledText("chat", "привет, как дела?")
  );

  private static final List<AdvisorRole> ADVISOR_ROLES = List.of(
    new AdvisorRole("Practical", SampleAdvisorPrompts.ROLE_PRACTICAL),
    new AdvisorRole("Abstract", SampleAdvisorPrompts.ROLE_ABSTRACT),
    new AdvisorRole("Consequence", SampleAdvisorPrompts.ROLE_CONSEQUENCE)
  );

  private Example() {
  }

  /**
   * Terminal flow: model → RAG mode → advisor count → load → session.
   * Embedding checkpoints skip RAG and advisors, then {@link #selectEncoderSession}.
   */
  public static void main(final String[] args) throws Exception {
    OrderedConsole console = new OrderedConsole(System.out, System.err);
    try (BufferedReader in = new BufferedReader(
      new InputStreamReader(System.in, StandardCharsets.UTF_8))) {

      boolean debug = hasFlag(args, DEBUG_FLAG);
      String[] rest = stripFlag(args, DEBUG_FLAG);

      Path modelPath = selectModel(rest, in, console);
      if (modelPath == null) {
        return;
      }

      Optional<RagMode> ragMode = Optional.of(RagMode.NONE);
      Optional<Integer> advisorCount = Optional.of(0);
      if (!ModelSupport.isEmbeddingCheckpoint(modelPath)
        && !ModelSupport.isSpeechCheckpoint(modelPath)) {
        ragMode = selectRagMode(in, console);
        if (ragMode.isEmpty()) {
          return;
        }
        advisorCount = selectAdvisorCount(in, console);
        if (advisorCount.isEmpty()) {
          return;
        }
      }

      LlmListener status = loadStatusListener(console);
      printLoadHints(modelPath, console);
      if (debug) {
        console.printlnInfo("Debug on: prepared model-user prompts print on stderr.");
      }

      try (LlmModel model = LlmModelFactory.make(modelPath, status)) {
        printSupportedModalities(model, console);
        if (model.isSpeechModel()) {
          runTranscribeSession(model, in, console);
          return;
        }
        if (model.isEmbeddingModel()) {
          Optional<EncoderSession> encoderSession = selectEncoderSession(in, console);
          if (encoderSession.isEmpty()) {
            return;
          }
          switch (encoderSession.get()) {
            case EMBED -> runEmbeddingSession(model, in, console);
            case CLASSIFY -> runClassificationSession(model, in, console);
          }
          return;
        }

        switch (ragMode.get()) {
          case NONE -> runPlainChatSession(model, advisorCount.get(), debug, status, in, console);
          case BM25 -> runBm25RagSession(model, advisorCount.get(), debug, status, in, console);
          case DENSE -> runDenseRagSession(model, advisorCount.get(), debug, status, in, console);
          case HYBRID -> runHybridRagSession(model, advisorCount.get(), debug, status, in, console);
        }
      }
    }
  }

  /**
   * Step 1 — choose a bundled checkpoint, or honor an explicit CLI / property / env path.
   *
   * <p>Resolution when a path is already given: first argument, then {@code -Dnanollvm.model},
   * then {@code NANOLLVM_MODEL}. Otherwise the menu lists downloaded checkpoints first (Qwen3-0.6B
   * preferred for chat quality). Enter selects item 1. If nothing is on disk, the demo exits with
   * download instructions.
   *
   * @return the path to load, or {@code null} if the user exits or no model is available
   */
  static Path selectModel(
    final String[] args,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    String[] rest = stripFlag(args, DEBUG_FLAG);
    if (hasExplicitModel(rest)) {
      Path requested = BundledModels.resolveDefault(rest);
      Optional<Path> found = BundledModels.find(requested.toString());
      if (found.isEmpty()) {
        console.println("Model not found: " + requested);
        printNoBundledModels(console);
        return null;
      }
      return found.get();
    }

    List<ModelChoice> menu = menuModels();
    List<ModelChoice> available = menu.stream()
      .filter(choice -> !choice.missing())
      .toList();
    if (available.isEmpty()) {
      printNoBundledModels(console);
      return null;
    }

    int exitNumber = menu.size() + 1;
    while (true) {
      console.println("Select model to load:");
      console.println(
        "  Kind: chat = instruct Q&A · base = plain completion (not chat-tuned)"
          + " · embeddings = vectors · speech = audio→text");
      for (int i = 0; i < menu.size(); i++) {
        console.println("  " + (i + 1) + ") " + menu.get(i).display());
      }
      console.println("  " + exitNumber + ") Exit");
      console.print("Choice [1-" + exitNumber + ", Enter=1]: ");

      String line = in.readLine();
      if (line == null) {
        return null;
      }

      String choice = line.strip();
      if (choice.isEmpty()) {
        ModelChoice first = available.getFirst();
        console.printlnInfo("Using " + first.label() + ".");
        return first.requirePath();
      }
      if (isExitChoice(choice, exitNumber)) {
        console.println("Bye.");
        return null;
      }

      try {
        int index = Integer.parseInt(choice) - 1;
        if (index >= 0 && index < menu.size()) {
          ModelChoice picked = menu.get(index);
          if (picked.missing()) {
            console.println(picked.missingHint());
            continue;
          }
          return picked.requirePath();
        }
      } catch (NumberFormatException ignored) {
        console.println("Enter 1 … " + exitNumber + ", or press Enter for 1.");
        continue;
      }

      console.println("Enter 1 … " + exitNumber + ", or press Enter for 1.");
    }
  }

  /**
   * Step 2 — choose how (or whether) the local {@code rag/} corpus grounds each turn.
   *
   * <p>{@link RagMode#NONE} is always available. BM25 needs a corpus folder. Dense and hybrid also
   * need a BERT-encoder checkpoint under {@code models/} (smallest matching GGUF or ONNX folder).
   *
   * @return the chosen mode, or empty if the user exits
   */
  static Optional<RagMode> selectRagMode(
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    boolean corpusPresent = BundledRag.find().isPresent();
    Optional<Path> ragEncoder = findRagEmbeddingModel();
    String encoderLabel = ragEncoder.map(Example::ragEmbeddingLabel)
      .orElse("BERT encoder under models/");

    while (true) {
      console.println("Select RAG index and use mode:");
      console.println("  1) None (plain chat)");
      console.println("  2) BM25 lexical" + missingMark(!corpusPresent, "no corpus at "
        + BundledRag.ragRoot()));
      console.println("  3) Dense embeddings (" + encoderLabel + ")"
        + missingMark(!corpusPresent, "no corpus")
        + missingMark(ragEncoder.isEmpty(), "no embedding model"));
      console.println("  4) Hybrid BM25 + dense"
        + missingMark(!corpusPresent, "no corpus")
        + missingMark(ragEncoder.isEmpty(), "no embedding model"));
      console.println("  5) Exit");
      console.print("Choice [1-5, Enter=1]: ");

      String line = in.readLine();
      if (line == null) {
        return Optional.empty();
      }

      String choice = line.strip();
      if (choice.isEmpty()) {
        choice = "1";
      }
      switch (choice) {
        case "1" -> {
          console.printlnInfo("RAG: off — plain chat.");
          return Optional.of(RagMode.NONE);
        }
        case "2" -> {
          if (!corpusPresent) {
            console.println("No RAG corpus. Create rag/ or set -Dnanollvm.rag.dir=…");
            continue;
          }
          return Optional.of(RagMode.BM25);
        }
        case "3" -> {
          if (!corpusPresent || ragEncoder.isEmpty()) {
            console.println(denseHybridHint(corpusPresent, ragEncoder.isPresent()));
            continue;
          }
          return Optional.of(RagMode.DENSE);
        }
        case "4" -> {
          if (!corpusPresent || ragEncoder.isEmpty()) {
            console.println(denseHybridHint(corpusPresent, ragEncoder.isPresent()));
            continue;
          }
          return Optional.of(RagMode.HYBRID);
        }
        case "5", "q", "quit", "exit" -> {
          console.println("Bye.");
          return Optional.empty();
        }
        default -> console.println("Enter 1, 2, 3, 4, or 5, or press Enter for 1.");
      }
    }
  }

  /**
   * Step 3 — how many named advisors run before each user turn ({@code 0} = off).
   *
   * <p>The sample ships three roles in order: Practical, Abstract, Consequence. The engine runs
   * them as one batched {@link LLM#generate} and the default mixer folds useful notes into the
   * main prompt.
   *
   * @return count {@code 0..3}, or empty if the user exits
   */
  static Optional<Integer> selectAdvisorCount(
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    while (true) {
      console.println("How many advisors to use?");
      console.println("  0) None");
      for (int count = 1; count <= ADVISOR_ROLES.size(); count++) {
        String names = ADVISOR_ROLES.stream()
          .limit(count)
          .map(AdvisorRole::name)
          .collect(joining(", "));
        console.println("  " + count + ") " + names);
      }
      console.println("  " + (ADVISOR_ROLES.size() + 1) + ") Exit");
      console.print("Choice [0-" + (ADVISOR_ROLES.size() + 1) + ", Enter=0]: ");

      String line = in.readLine();
      if (line == null) {
        return Optional.empty();
      }

      String choice = line.strip();
      if (choice.isEmpty()) {
        return Optional.of(0);
      }
      if (isExitChoice(choice, ADVISOR_ROLES.size() + 1)) {
        console.println("Bye.");
        return Optional.empty();
      }

      try {
        int count = Integer.parseInt(choice);
        if (count >= 0 && count <= ADVISOR_ROLES.size()) {
          return Optional.of(count);
        }
      } catch (NumberFormatException ignored) {
        console.println("Enter 0 … " + (ADVISOR_ROLES.size() + 1) + ", or press Enter for 0.");
        continue;
      }

      console.println("Enter 0 … " + (ADVISOR_ROLES.size() + 1) + ", or press Enter for 0.");
    }
  }

  /**
   * Speech REPL: each non-empty line is a WAV path passed to {@link LlmModel#transcribe(Path)}.
   *
   * <p>{@code /tone} writes a short 440 Hz beep and transcribes it. {@code /lang xx} sets a
   * language hint; {@code /lang auto} clears it. Whisper checkpoints are not chat models.
   */
  private static void runTranscribeSession(
    final LlmModel model,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    console.printlnInfo(
      "Speech model (" + model.architectureName()
        + ", " + model.modalities() +
        ") — enter a WAV path. Commands: /tone  /lang <code>|auto  /exit");
    console.println();

    String language = null;
    while (true) {
      console.print("wav?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        return;
      }

      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isQuitCommand(user)) {
        return;
      }
      if (user.equalsIgnoreCase("/tone")) {
        Path tone = Files.createTempFile("nanollvm-example-tone-", ".wav");
        Files.write(tone, toneWavBytes());
        transcribePath(model, tone, language, console);
        continue;
      }
      if (user.regionMatches(true, 0, "/lang ", 0, 6)) {
        String code = user.substring(6).strip();
        language = code.isEmpty() || "auto".equalsIgnoreCase(code) ? null : code;
        console.println(language == null ? "language=auto" : "language=" + language);
        continue;
      }

      transcribePath(model, Path.of(user), language, console);
    }
  }

  private static void transcribePath(
    final LlmModel model,
    final Path wav,
    final String language,
    final OrderedConsole console
  ) {
    try {
      long started = System.nanoTime();
      String text = language == null
        ? model.transcribe(wav)
        : model.transcribe(wav, language);
      console.printf(
        Locale.ROOT,
        "%.3fs  %s%n%n",
        (System.nanoTime() - started) / 1e9,
        text.isBlank() ? "(empty)" : text);
    } catch (Exception ex) {
      console.println("transcribe failed: " + ex.getMessage());
      console.println();
    }
  }

  private static byte[] toneWavBytes() {
    int n = 8_000;
    ByteBuffer pcm = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN);
    pcm.put("RIFF".getBytes(StandardCharsets.US_ASCII));
    pcm.putInt(36 + n * 2);
    pcm.put("WAVE".getBytes(StandardCharsets.US_ASCII));
    pcm.put("fmt ".getBytes(StandardCharsets.US_ASCII));
    pcm.putInt(16);
    pcm.putShort((short) 1);
    pcm.putShort((short) 1);
    pcm.putInt(16_000);
    pcm.putInt(32_000);
    pcm.putShort((short) 2);
    pcm.putShort((short) 16);
    pcm.put("data".getBytes(StandardCharsets.US_ASCII));
    pcm.putInt(n * 2);
    for (int i = 0; i < n; i++) {
      double sample = 0.2 * Math.sin(2.0 * Math.PI * 440.0 * i / 16_000.0);
      pcm.putShort((short) Math.round(sample * 32767.0));
    }
    return pcm.array();
  }

  /**
   * Embedding REPL: each non-empty line is {@link LlmModel#embed(CharSequence)}.
   *
   * <p>BERT-encoder checkpoints are not chat models — never {@code LLM.builder}. Prints dimension, L2 norm, a short preview, and
   * cosine vs the previous vector.
   */
  private static void runEmbeddingSession(
    final LlmModel model,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    console.printlnInfo(
      "Embedding model (" + model.architectureName()
        + ", " + model.modalities() + ") — each line → L2-normalized vector.");
    console.println("Type text and press Enter. Commands: /exit  /quit  /clear");
    console.println("For labels, restart and choose Classify after the model loads.");
    console.println();

    float[] previous = null;
    while (true) {
      console.print("embed?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        return;
      }

      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isQuitCommand(user)) {
        return;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        previous = null;
        console.println("(previous embedding cleared)");
        continue;
      }

      long started = System.nanoTime();
      float[] vector = model.embed(user);
      console.printf(
        Locale.ROOT,
        "dim=%d  L2=%.4f  %.3fs  preview=%s%n",
        vector.length,
        l2Norm(vector),
        (System.nanoTime() - started) / 1e9,
        preview(vector));
      if (previous != null) {
        console.printf(Locale.ROOT, "cos(prev)=%.4f%n", cosine(previous, vector));
      }
      previous = vector;
      console.println();
    }
  }

  /**
   * After an embedding checkpoint loads: vectors, or a few-shot label probe on those vectors.
   *
   * @return the session, or empty if the user exits
   */
  static Optional<EncoderSession> selectEncoderSession(
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    while (true) {
      console.println("Select encoder session:");
      console.println("  1) Embed (vectors + cosine)");
      console.println("  2) Classify (few-shot labels; use this for XLM-RoBERTa)");
      console.println("  3) Exit");
      console.print("Choice [1-3, Enter=1]: ");

      String line = in.readLine();
      if (line == null) {
        return Optional.empty();
      }

      String choice = line.strip();
      if (choice.isEmpty()) {
        choice = "1";
      }
      switch (choice) {
        case "1" -> {
          console.printlnInfo("Encoder: embed.");
          return Optional.of(EncoderSession.EMBED);
        }
        case "2" -> {
          console.printlnInfo("Encoder: classify.");
          return Optional.of(EncoderSession.CLASSIFY);
        }
        case "3", "q", "quit", "exit" -> {
          console.println("Bye.");
          return Optional.empty();
        }
        default -> console.println("Enter 1, 2, or 3, or press Enter for 1.");
      }
    }
  }

  /**
   * Few-shot classify: teach {@code label | text} (or {@code /load} a TSV), then unlabeled lines
   * predict. Uses {@link EmbeddingClassifier} on {@link LlmModel#embed} — not a classification head.
   */
  private static void runClassificationSession(
    final LlmModel model,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    console.printlnInfo(
      "Classify with " + model.architectureName()
        + " vectors (centered prototypes, not a Hub classification head).");
    printClassifyHelp(console);
    console.println();

    EmbeddingClassifier.Trainer trainer = EmbeddingClassifier.trainer();
    List<LabeledText> taught = new ArrayList<>();
    EmbeddingClassifier fitted = null;

    while (true) {
      console.print(fitted == null ? "teach?> " : "classify?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        return;
      }

      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isQuitCommand(user)) {
        return;
      }

      Optional<ClassifyCommand> command = ClassifyCommand.parse(user);
      if (command.isPresent()) {
        switch (command.get()) {
          case HELP -> printClassifyHelp(console);
          case FORGET -> {
            trainer.clear();
            taught.clear();
            fitted = null;
            console.println("(examples cleared)");
          }
          case LABELS -> printTaughtLabels(taught, console);
          case DEMO -> {
            fitted = teachAll(CLASSIFY_DEMO, model, trainer, taught, console);
          }
          case LOAD -> fitted = loadLabelFile(user, model, trainer, taught, console).orElse(fitted);
        }
        continue;
      }

      Optional<LabeledText> labeled = EmbeddingClassifier.parseLabeledLine(user);
      if (labeled.isPresent()) {
        fitted = teachOne(labeled.get(), model, trainer, taught, console);
        continue;
      }

      if (fitted == null) {
        console.println("Need two labels first. Example:  poem | frosty winter   or  /demo");
        continue;
      }
      long started = System.nanoTime();
      printPrediction(fitted, model.embed(user), (System.nanoTime() - started) / 1e9, console);
      console.println();
    }
  }

  private static void printClassifyHelp(final OrderedConsole console) {
    console.println(
      "Teach:  label | text     or tab-separated. Then type unlabeled text to predict.");
    console.println("Commands: /demo  /load <file>  /labels  /forget  /help  /exit  /quit");
    console.println("Example file (from repo root):  /load " + CLASSIFY_LABELS_EXAMPLE);
  }

  private static void printTaughtLabels(
    final List<LabeledText> taught,
    final OrderedConsole console
  ) {
    if (taught.isEmpty()) {
      console.println("(no examples yet)");
      return;
    }
    taught.forEach(example ->
      console.println(example.label() + " | " + example.text()));
  }

  private static Optional<EmbeddingClassifier> loadLabelFile(
    final String user,
    final LlmModel model,
    final EmbeddingClassifier.Trainer trainer,
    final List<LabeledText> taught,
    final OrderedConsole console
  ) {
    String pathText = user.length() > 5 ? user.substring(5).strip() : "";
    if (pathText.isEmpty()) {
      console.println("Usage: /load path/to/labels.txt");
      return Optional.empty();
    }
    Path path = Path.of(pathText);
    if (!Files.isRegularFile(path)) {
      console.println("File not found: " + path.toAbsolutePath().normalize());
      return Optional.empty();
    }
    try {
      List<LabeledText> loaded = Files.readAllLines(path, StandardCharsets.UTF_8).stream()
        .map(EmbeddingClassifier::parseLabeledLine)
        .flatMap(Optional::stream)
        .toList();
      if (loaded.isEmpty()) {
        console.println("No labeled lines in " + path);
        return Optional.empty();
      }
      return Optional.ofNullable(teachAll(loaded, model, trainer, taught, console));
    } catch (IOException e) {
      console.println("Could not read " + path + ": " + e.getMessage());
      return Optional.empty();
    }
  }

  private static EmbeddingClassifier teachAll(
    final List<LabeledText> examples,
    final LlmModel model,
    final EmbeddingClassifier.Trainer trainer,
    final List<LabeledText> taught,
    final OrderedConsole console
  ) {
    examples.forEach(example -> teachOne(example, model, trainer, taught, console));
    return trainer.canFit() ? trainer.fit() : null;
  }

  private static EmbeddingClassifier teachOne(
    final LabeledText example,
    final LlmModel model,
    final EmbeddingClassifier.Trainer trainer,
    final List<LabeledText> taught,
    final OrderedConsole console
  ) {
    long started = System.nanoTime();
    trainer.add(example.label(), model.embed(example.text()));
    taught.add(example);
    EmbeddingClassifier fitted = trainer.canFit() ? trainer.fit() : null;
    console.printf(
      Locale.ROOT,
      "+ %s  (%d examples, %d labels, %.3fs)%n",
      example.label(),
      trainer.size(),
      trainer.labels().size(),
      (System.nanoTime() - started) / 1e9);
    if (fitted == null) {
      console.println("  add another label to start predicting");
    }
    return fitted;
  }

  private static void printPrediction(
    final EmbeddingClassifier classifier,
    final float[] vector,
    final double seconds,
    final OrderedConsole console
  ) {
    Prediction prediction = classifier.classify(vector);
    console.printf(
      Locale.ROOT,
      "%s%s  %.3fs%n",
      prediction.label(),
      prediction.isClose(CLASSIFY_CLOSE_MARGIN) ? "  (close scores — add more examples)" : "",
      seconds);
    prediction.scores().forEach(score ->
      console.printf(Locale.ROOT, "  %-12s  %.4f%n", score.label(), score.cosine()));
  }

  /**
   * Plain chat: {@code LLM.builder} → {@link LLM#chat} → {@link ChatSession#send}.
   *
   * <p>No document retrieval. Answers come from weights plus conversation history only.
   */
  private static void runPlainChatSession(
    final LlmModel model,
    final int advisorCount,
    final boolean debug,
    final LlmListener status,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    try (LLM llm = openEngine(model, advisorCount, status, console)) {
      ChatSession chat = openChatSession(llm, debug);
      chat.streamTo(thinkStream(console), answerStream(console), useColor());
      runConversation(
        in,
        console,
        "?> ",
        chat::send,
        chat::clear,
        "(conversation cleared)",
        null);
    }
  }

  /**
   * BM25 RAG: {@link RagFactory#tryMake(Path, RagLoadOptions, LlmListener)} indexes {@code rag/}
   * once; each turn {@link LLM#rag(RagIndex, int)} retrieves chunks by term overlap and places
   * them in the user prompt before generate.
   */
  private static void runBm25RagSession(
    final LlmModel model,
    final int advisorCount,
    final boolean debug,
    final LlmListener status,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    Optional<PreparedRag> index = prepareLexicalIndex(model, status, console);
    if (index.isEmpty()) {
      runPlainChatSession(model, advisorCount, debug, status, in, console);
      return;
    }

    console.printlnInfo(
      "RAG: BM25 over " + BundledRag.ragRoot() + " (" + index.get().size() + " chunks)");

    try (LLM llm = openEngine(model, advisorCount, status, console)) {
      converseWithRag(llm, index.get(), debug, in, console);
    }
  }

  /**
   * Dense RAG: embed every BM25 chunk with the bundled encoder, retrieve by cosine similarity.
   *
   * <p>{@link DenseRagIndex#of(PreparedRag, LlmModel)} builds passage vectors; {@link LLM#rag}
   * then ranks the question against those vectors. The embedding model is a second
   * {@link LlmModel} — not {@code LLM.builder}.
   */
  private static void runDenseRagSession(
    final LlmModel model,
    final int advisorCount,
    final boolean debug,
    final LlmListener status,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    Optional<PreparedRag> lexical = prepareLexicalIndex(model, status, console);
    Path encoderPath = findRagEmbeddingModel().orElse(null);
    if (lexical.isEmpty() || encoderPath == null) {
      console.printlnInfo("Dense RAG unavailable — falling back to plain chat.");
      runPlainChatSession(model, advisorCount, debug, status, in, console);
      return;
    }

    console.printlnInfo("Loading RAG embedding model from " + encoderPath);
    try (LlmModel embed = LlmModelFactory.make(encoderPath, status)) {
      DenseRagIndex index = DenseRagIndex.of(lexical.get(), embed);
      console.printlnInfo(
        "RAG: dense embeddings over " + BundledRag.ragRoot()
          + " (" + index.size() + " chunks; encoder " + ragEmbeddingLabel(encoderPath)
          + " / " + embed.architectureName() + ")");

      try (LLM llm = openEngine(model, advisorCount, status, console)) {
        converseWithRag(llm, index, debug, in, console);
      }
    }
  }

  /**
   * Hybrid RAG: {@link RagFactory#withEmbeddings(PreparedRag, LlmModel)} fuses BM25 ranks with
   * dense cosine ranks (reciprocal rank fusion), then the same {@link RagSession} path as BM25.
   */
  private static void runHybridRagSession(
    final LlmModel model,
    final int advisorCount,
    final boolean debug,
    final LlmListener status,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    Optional<PreparedRag> lexical = prepareLexicalIndex(model, status, console);
    Path encoderPath = findRagEmbeddingModel().orElse(null);
    if (lexical.isEmpty() || encoderPath == null) {
      console.printlnInfo("Hybrid RAG unavailable — falling back to plain chat.");
      runPlainChatSession(model, advisorCount, debug, status, in, console);
      return;
    }

    console.printlnInfo("Loading RAG embedding model from " + encoderPath);
    try (LlmModel embed = LlmModelFactory.make(encoderPath, status)) {
      RagIndex index = RagFactory.withEmbeddings(lexical.get(), embed);
      console.printlnInfo(
        "RAG: hybrid BM25+dense over " + BundledRag.ragRoot()
          + " (" + index.size() + " chunks; encoder " + ragEmbeddingLabel(encoderPath)
          + " / " + embed.architectureName() + ")");

      try (LLM llm = openEngine(model, advisorCount, status, console)) {
        converseWithRag(llm, index, debug, in, console);
      }
    }
  }

  /**
   * Builds one {@link LLM} on the loaded chat model.
   *
   * <p>Advisor count {@code 0} skips {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)}. Counts {@code 1..3} take
   * Practical / Abstract / Consequence in that order, plus a note filter that drops short
   * setup-boilerplate advisor lines.
   */
  private static LLM openEngine(
    final LlmModel model,
    final int advisorCount,
    final LlmListener status,
    final OrderedConsole console
  ) {
    LLM.Builder builder = LLM.builder(model)
      .maxNumSeqs(4)
      .maxModelLen(2048)
      .listen(status);

    String system = SampleChatPrompts.forDemo(model.architectureName(), model.tokenizer());
    List<LlmAdvisor> advisors = advisorsForCount(advisorCount);
    if (!advisors.isEmpty()) {
      builder.advisors(LlmAdvisorMixer.defaults(), advisors);
      builder.advisorNoteFilter(note -> !SampleChatPrompts.isSetupBoilerplate(note));
      system = SampleAdvisorPrompts.withAdvisorAddon(system);
      console.printlnInfo(
        "Advisors: " + advisors.stream().map(LlmAdvisor::name).collect(joining(", ")) + ".");
    } else {
      console.printlnInfo("Advisors: off.");
    }

    return builder.systemPrompt(system).build();
  }

  /**
   * Opens a {@link ChatSession} with demo sampling. Turn-based tokenizers (e.g. Gemma) also enable
   * unusable-answer recovery so short setup acknowledgments are retried.
   *
   * <p>Prepared-prompt {@code debug>} lines are off unless {@code --debug} was passed to
   * {@link #main}.
   */
  private static ChatSession openChatSession(final LLM llm, final boolean debug) {
    ChatSession chat = llm.chat(
      SampleChatPrompts.samplingForDemo(llm.tokenizer(), maxNewTokens(llm)));
    chat.emitDebugPrompts(debug);
    applyTurnBasedRecovery(chat, llm.tokenizer().isTurnBasedChat());
    return chat;
  }

  /**
   * Opens a {@link RagSession}: retrieve → format passages into the user turn → chat generate.
   */
  private static RagSession openRagSession(
    final LLM llm,
    final RagIndex index,
    final boolean debug
  ) {
    boolean turnBased = llm.tokenizer().isTurnBasedChat();
    int maxTokens = ragAnswerTokens(llm, turnBased);

    RagSession rag = llm.rag(index, maxTokens)
      .maxTokensWhenNoHits(ragNoHitTokens(llm, maxTokens))
      .topK(turnBased ? RAG_TOP_K_TURN_BASED : RAG_TOP_K_DEFAULT)
      .maxContextChars(ragContextChars(llm, turnBased, maxTokens))
      .isolateGeneration(turnBased)
      .enableThinking(llm.tokenizer().invitesThinking())
      .sampling(SamplingParams.builder()
        .temperature(turnBased ? 0.1f : 0.4f)
        .maxTokens(maxTokens)
        .topK(turnBased ? 30 : 0)
        .topP(turnBased ? 0.8f : 0.85f)
        .build());

    rag.emitDebugPrompts(debug);
    applyTurnBasedRecovery(rag, turnBased);
    return rag;
  }

  private static void converseWithRag(
    final LLM llm,
    final RagIndex index,
    final boolean debug,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    console.printlnInfo("Ask about the docs in rag/ (engine, models, Nile, capitals, …).");

    RagSession rag = openRagSession(llm, index, debug);
    rag.streamTo(thinkStream(console), answerStream(console), useColor());
    runConversation(
      in,
      console,
      "rag?> ",
      rag::send,
      rag::clear,
      "(conversation cleared; RAG index kept)",
      rag::lastHits);
  }

  private static Optional<PreparedRag> prepareLexicalIndex(
    final LlmModel chatModel,
    final LlmListener status,
    final OrderedConsole console
  ) {
    Optional<Path> ragRoot = BundledRag.find();
    if (ragRoot.isEmpty()) {
      console.printlnInfo("RAG: no corpus at " + BundledRag.ragRoot() + " — plain chat.");
      return Optional.empty();
    }

    console.printlnInfo("Preparing RAG corpus from " + ragRoot.get());
    RagLoadOptions options = usesTinyRagChunks(chatModel)
      ? RagLoadOptions.forTinyModels()
      : RagLoadOptions.defaults();
    Optional<PreparedRag> index = RagFactory.tryMake(ragRoot.get(), options, status);
    if (index.isEmpty()) {
      console.printlnInfo("RAG: no documents in " + ragRoot.get() + " — plain chat.");
    }
    return index;
  }

  private static void runConversation(
    final BufferedReader in,
    final OrderedConsole console,
    final String promptLabel,
    final Function<String, ChatReply> send,
    final Runnable clear,
    final String clearedMessage,
    final Supplier<List<RagHit>> lastHits
  ) throws Exception {
    console.println("Type a message and press Enter. Commands: /exit  /quit  /clear");
    console.println(
      "Answer/prompts on stdout; thinking and load/status on stderr (red in many IDEs).");
    console.println("Prepared prompts (debug>) only with --debug.");
    console.println(
      "After each turn: engine tok/s from GenerationStats (main generate; excludes advisors / RAG prep).");
    console.println();

    long totalTokens = 0;
    long totalNanos = 0;
    int turns = 0;

    while (true) {
      console.print(promptLabel);
      String line = in.readLine();
      if (line == null) {
        console.println();
        break;
      }

      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isQuitCommand(user)) {
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        clear.run();
        console.println(clearedMessage);
        continue;
      }

      ChatReply reply = send.apply(user);
      int tokens = reply.stats().completionTokens();
      long nanos = Math.max(1L, reply.stats().elapsedNanos());
      totalTokens += tokens;
      totalNanos += nanos;
      turns++;
      console.printf(
        Locale.ROOT,
        "(turn %d: %d tok in %.2fs → %.1f tok/s; session avg %.1f tok/s)%n",
        turns,
        tokens,
        nanos / 1e9,
        reply.stats().completionTokensPerSecond(),
        totalTokens / (totalNanos / 1e9));

      if (lastHits != null) {
        printRagHits(console, lastHits.get());
      }
      console.println();
    }

    if (turns > 0) {
      console.printf(
        Locale.ROOT,
        "(session: %d turn(s), %d tok, %.2fs → avg %.1f tok/s)%n",
        turns,
        totalTokens,
        totalNanos / 1e9,
        totalTokens / (totalNanos / 1e9));
    }
  }

  private static void printRagHits(final OrderedConsole console, final List<RagHit> hits) {
    if (hits == null || hits.isEmpty()) {
      console.println("(no RAG hits)");
      return;
    }

    console.println("(retrieved " + hits.size() + " chunk(s))");
    int index = 1;
    for (RagHit hit : hits) {
      console.printf(Locale.ROOT, "  [%d] %.3f  %s%n",
        index++, hit.score(), ragHitSourceName(hit));
      console.println("      " + hit.chunk().text().strip());
    }
  }

  private static String ragHitSourceName(final RagHit hit) {
    String source = hit.chunk().source();
    Path name = Path.of(source).getFileName();
    return name == null ? source : name.toString();
  }

  private static void applyTurnBasedRecovery(final ChatSession chat, final boolean turnBased) {
    if (!turnBased) {
      return;
    }
    chat.recoverUnusableAnswers(true)
      .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
      .unusableAnswerFallback("What would you like to explore?");
  }

  private static void applyTurnBasedRecovery(final RagSession rag, final boolean turnBased) {
    if (!turnBased) {
      return;
    }
    rag.recoverUnusableAnswers(true)
      .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
      .unusableAnswerFallback("What would you like to explore?");
  }

  /**
   * Bundled checkpoints in preference order: Qwen3 chat first, then Gemma, large GGUF, compact
   * ONNX demos, embeddings.
   */
  static List<ModelChoice> catalog() {
    return List.of(
      choice(
        "Qwen3-0.6B (chat, safetensors)",
        BundledModels.QWEN3_0_6B,
        "Run models/download-qwen3-0.6b.sh"),
      choice(
        "Gemma3-270M (chat, safetensors)",
        BundledModels.GEMMA3_270M,
        "Run models/download-gemma3-270m.sh (HF license + HF_TOKEN)"),
      choice(
        "Gemma4-E2B QAT mobile (chat, safetensors, ~2.3GB)",
        BundledModels.GEMMA4_E2B_IT_QAT_MOBILE,
        "Run models/download-gemma4-e2b-qat-mobile.sh"),
      choice(
        "LFM2.5-2.6B Q4_K_M (chat, gguf, ~16g heap)",
        BundledModels.LFM2_5_2_6B_GGUF,
        "Run models/download-lfm2.5-2.6b-gguf.sh"),
      choice(
        "SmolLM2-135M-Instruct-ONNX (compact onnx demo)",
        BundledModels.SMOLLM2_135M_INSTRUCT_ONNX,
        "Run models/download-smollm2-135m-instruct-onnx.sh"),
      choice(
        "Tiny-LLM-ONNX (base, onnx ~10M)",
        BundledModels.TINY_LLM_ONNX,
        "Run models/download-tiny-llm-onnx.sh"),
      choice(
        "gte-small Q2_K (embeddings, gguf)",
        BundledModels.GTE_SMALL_GGUF,
        "Run models/download-gte-small-gguf.sh"),
      choice(
        "multilingual-e5-small (embeddings, onnx)",
        BundledModels.MULTILINGUAL_E5_SMALL,
        "Run models/download-multilingual-e5-small.sh"),
      choice(
        "xlm-roberta-base (embeddings, onnx, ~1.9GB)",
        BundledModels.XLM_ROBERTA_BASE,
        "Run models/download-xlm-roberta-base.sh"),
      choice(
        "whisper-base (speech, safetensors, ~290MB)",
        BundledModels.WHISPER_BASE,
        "Run models/download-whisper-base.sh"),
      choice(
        "whisper-tiny (speech, safetensors, ~150MB)",
        BundledModels.WHISPER_TINY,
        "Run models/download-whisper-tiny.sh")
    );
  }

  /**
   * Downloaded checkpoints first (same preference order), then missing ones marked
   * {@code [not downloaded]}.
   */
  static List<ModelChoice> menuModels() {
    List<ModelChoice> all = catalog();
    return Stream.concat(
      all.stream().filter(choice -> !choice.missing()),
      all.stream().filter(ModelChoice::missing)
    ).toList();
  }

  static Path resolveModel(final String[] args, final BufferedReader in) throws Exception {
    return selectModel(args, in, new OrderedConsole(System.out, System.err));
  }

  static List<LlmAdvisor> advisorsForCount(final int count) {
    return ADVISOR_ROLES.stream()
      .limit(Math.clamp(count, 0, ADVISOR_ROLES.size()))
      .map(AdvisorRole::toAdvisor)
      .toList();
  }

  private static ModelChoice choice(
    final String label,
    final String bundledPath,
    final String missingHint
  ) {
    Optional<Path> path = BundledModels.find(bundledPath);
    return new ModelChoice(label, path, missingHint, path.isEmpty());
  }

  private static boolean hasExplicitModel(final String[] args) {
    return (args != null && args.length > 0 && args[0] != null && !args[0].isBlank())
      || NanoLlvmProps.systemProperty(PROP_MODEL) != null
      || NanoLlvmProps.environment(ENV_MODEL) != null;
  }

  private static boolean hasFlag(final String[] args, final String flag) {
    if (args == null || flag == null) {
      return false;
    }
    return Arrays.stream(args).anyMatch(flag::equalsIgnoreCase);
  }

  private static String[] stripFlag(final String[] args, final String flag) {
    if (args == null || args.length == 0) {
      return new String[0];
    }
    return Arrays.stream(args)
      .filter(arg -> !flag.equalsIgnoreCase(arg))
      .toArray(String[]::new);
  }

  private static void printNoBundledModels(final OrderedConsole console) {
    console.println("No usable model found under " + BundledModels.modelsRoot() + ".");
    console.println("Download a chat checkpoint (recommended) and retry:");
    console.println();
    console.println("  ./models/download-qwen3-0.6b.sh");
    console.println();
    console.println("Other checkpoints:");
    console.println("  ./models/download-gemma3-270m.sh      (HF license + HF_TOKEN)");
    console.println("  ./models/download-gemma4-e2b-qat-mobile.sh  (~2.3GB, Apache 2.0)");
    console.println("  ./models/download-lfm2.5-2.6b-gguf.sh");
    console.println("  ./models/download-smollm2-135m-instruct-onnx.sh  (compact onnx demo)");
    console.println("  ./models/download-tiny-llm-onnx.sh");
    console.println("  ./models/download-gte-small-gguf.sh    (embeddings)");
    console.println("  ./models/download-multilingual-e5-small.sh  (multilingual embeddings)");
    console.println("  ./models/download-xlm-roberta-base.sh  (XLM-RoBERTa ONNX embeddings)");
    console.println("  ./models/download-whisper-base.sh      (speech-to-text, ~290MB)");
    console.println("  ./models/download-whisper-tiny.sh      (smaller speech-to-text, ~150MB)");
    console.println();
    console.println("Windows: matching .ps1 / .cmd scripts in models/.");
    console.println(
      "Or pass a local folder: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.args=models/YourModel");
  }

  private static void printSupportedModalities(
    final LlmModel model,
    final OrderedConsole console
  ) {
    console.printlnInfo(
      "Checkpoint modalities: input %s, output %s.".formatted(
        formatModalities(model.inputModalities()),
        formatModalities(model.outputModalities())));
    if (!model.modalities().equals(model.usableModalities())) {
      console.printlnInfo(
        "This library runs: input %s, output %s.".formatted(
          formatModalities(model.usableModalities().input()),
          formatModalities(model.usableModalities().output())));
    }
  }

  private static String formatModalities(final Set<LlmModality> modalities) {
    return modalities.isEmpty()
      ? "none"
      : modalities.stream().map(LlmModality::wireName).collect(joining("+"));
  }

  private static void printLoadHints(final Path path, final OrderedConsole console) {
    console.printlnInfo("Loading model from " + path);
    console.printlnInfo(
      "Architecture auto-detects from config.json / GGUF metadata "
        + "(override: -Dnanollvm.arch=qwen3|gemma3|llama|lfm2|bert|whisper).");
    console.printlnInfo(
      "CPU matmul: " + Runtime.getRuntime().availableProcessors()
        + " threads from Runtime (override: -Dnanollvm.cpu.threads=N).");
    if (isTinyLlmPath(path)) {
      console.printlnInfo(
        "This checkpoint is a base/completion toy model (~10M), not chat-tuned — "
          + "expect odd replies under chat/RAG.");
    }
    if (isSmolLm2Path(path)) {
      console.printlnInfo(
        "SmolLM2 Instruct is a compact ONNX demo (~135M) — useful to smoke-test ONNX chat, "
          + "weaker dialog than Qwen3 / Gemma3.");
    }
    if (ModelSupport.isEmbeddingCheckpoint(path)) {
      console.printlnInfo(
        "This checkpoint is a BERT-family embedding encoder — LlmModel.embed, not LLM.builder.");
    }
    if (ModelSupport.isSpeechCheckpoint(path)) {
      console.printlnInfo(
        "This checkpoint is Whisper speech-to-text — LlmModel.transcribe, not LLM.builder.");
    }
    if (isGgufPath(path)) {
      console.printlnInfo(
        "GGUF: weights stay packed (dequant on matmul). For float32 at load: "
          + "LlmModelFactory.make(path, io, true).");
    }
  }

  private static LlmListener loadStatusListener(final OrderedConsole console) {
    return (source, event) -> {
      switch (event.kind()) {
        case STATUS_INFO -> console.printInfo(event.text());
        case STATUS_PROGRESS -> console.print(event.text());
        default -> {
        }
      }
    };
  }

  private static int maxNewTokens(final LLM llm) {
    int maxNew = Math.clamp(llm.config().maxModelLen() / 2, 32, MAX_NEW_TOKENS);
    return isCompactDemoModel(llm) ? Math.min(maxNew, COMPACT_DEMO_MAX_NEW_TOKENS) : maxNew;
  }

  private static int ragAnswerTokens(final LLM llm, final boolean turnBased) {
    int cap = turnBased ? RAG_MAX_TOKENS_TURN_BASED : RAG_MAX_TOKENS_DEFAULT;
    int maxTokens = Math.clamp(llm.config().maxModelLen() / 4, 32, cap);
    return isCompactDemoModel(llm) ? Math.min(maxTokens, COMPACT_DEMO_MAX_NEW_TOKENS) : maxTokens;
  }

  private static int ragNoHitTokens(final LLM llm, final int maxTokens) {
    int min = isCompactDemoModel(llm) ? maxTokens : llm.config().maxModelLen() / 2;
    int max = maxNewTokens(llm);
    return min > max ? max : Math.clamp(maxTokens, min, max);
  }

  private static int ragContextChars(
    final LLM llm,
    final boolean turnBased,
    final int maxTokens
  ) {
    int cap = turnBased ? RAG_CONTEXT_CHARS_TURN_BASED : RAG_CONTEXT_CHARS_DEFAULT;
    int chars = Math.clamp((llm.config().maxModelLen() - maxTokens - 64L) * 3L, 256, cap);
    return isCompactDemoModel(llm) ? Math.min(chars, 1200) : chars;
  }

  private static boolean usesTinyRagChunks(final LlmModel model) {
    return model.tokenizer().isTurnBasedChat() || isCompactDemoModel(model);
  }

  private static boolean isCompactDemoModel(final LLM llm) {
    return isCompactDemoModel(llm.config().hfConfig().numHiddenLayers(),
      llm.config().hfConfig().hiddenSize());
  }

  private static boolean isCompactDemoModel(final LlmModel model) {
    return isCompactDemoModel(model.hfConfig().numHiddenLayers(), model.hfConfig().hiddenSize());
  }

  private static boolean isCompactDemoModel(final int layers, final int hiddenSize) {
    return layers <= 32 && hiddenSize <= 768;
  }

  private static boolean isGgufPath(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  private static boolean isTinyLlmPath(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).contains("tiny-llm");
  }

  private static boolean isSmolLm2Path(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).contains("smollm2");
  }

  private static Optional<Path> findRagEmbeddingModel() {
    Path root = BundledModels.modelsRoot();
    if (!Files.isDirectory(root)) {
      return Optional.empty();
    }
    try (Stream<Path> children = Files.list(root)) {
      return children
        .flatMap(Example::checkpointPaths)
        .filter(ModelSupport::isEmbeddingCheckpoint)
        .min(Comparator.comparingLong(Example::checkpointWeightBytes));
    } catch (IOException ignored) {
      return Optional.empty();
    }
  }

  private static Stream<Path> checkpointPaths(final Path path) {
    if (Files.isRegularFile(path) && isGgufPath(path)) {
      return Stream.of(path);
    }
    if (!Files.isDirectory(path)) {
      return Stream.empty();
    }
    if (Files.isRegularFile(path.resolve("config.json"))) {
      return Stream.of(path);
    }
    try (Stream<Path> nested = Files.list(path)) {
      return nested.filter(p -> Files.isRegularFile(p) && isGgufPath(p)).toList().stream();
    } catch (IOException ignored) {
      return Stream.empty();
    }
  }

  private static String ragEmbeddingLabel(final Path path) {
    Path fileName = path.getFileName();
    return fileName == null ? path.toString() : fileName.toString();
  }

  private static long checkpointWeightBytes(final Path path) {
    try {
      if (Files.isRegularFile(path)) {
        return Files.size(path);
      }
      try (Stream<Path> walk = Files.walk(path, 2)) {
        return walk.filter(Example::isWeightFile)
          .mapToLong(Example::fileSizeOrMax)
          .min()
          .orElse(Long.MAX_VALUE);
      }
    } catch (IOException ignored) {
      return Long.MAX_VALUE;
    }
  }

  private static boolean isWeightFile(final Path path) {
    if (!Files.isRegularFile(path) || path.getFileName() == null) {
      return false;
    }
    String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
    return name.endsWith(".onnx") || name.endsWith(".gguf");
  }

  private static long fileSizeOrMax(final Path path) {
    try {
      return Files.size(path);
    } catch (IOException ignored) {
      return Long.MAX_VALUE;
    }
  }

  private static boolean isQuitCommand(final String user) {
    String command = user.toLowerCase(Locale.ROOT);
    return command.equals("/exit") || command.equals("/quit")
      || command.equals("exit") || command.equals("quit");
  }

  private static boolean isExitChoice(final String choice, final int exitNumber) {
    return choice.equals(Integer.toString(exitNumber))
      || choice.equalsIgnoreCase("q")
      || choice.equalsIgnoreCase("quit")
      || choice.equalsIgnoreCase("exit");
  }

  private static String missingMark(final boolean missing, final String reason) {
    return missing ? "  [" + reason + "]" : "";
  }

  private static String denseHybridHint(final boolean corpusPresent, final boolean encoderPresent) {
    if (!corpusPresent && !encoderPresent) {
      return "Need a rag/ corpus and a BERT-encoder checkpoint under models/ "
        + "(GGUF or ONNX; e.g. models/download-gte-small-gguf.sh)";
    }
    if (!corpusPresent) {
      return "No RAG corpus. Create rag/ or set -Dnanollvm.rag.dir=…";
    }
    return "No BERT-encoder checkpoint under models/. "
      + "Download a GGUF or ONNX encoder (e.g. models/download-gte-small-gguf.sh).";
  }

  private static boolean useColor() {
    return System.getenv("NO_COLOR") == null
      && !"false".equalsIgnoreCase(NanoLlvmProps.systemProperty(PROP_COLOR));
  }

  private static PrintStream thinkStream(final OrderedConsole console) {
    return console.infoStream();
  }

  private static PrintStream answerStream(final OrderedConsole console) {
    return console.stream();
  }

  private static double l2Norm(final float[] vector) {
    double sum = 0.0;
    for (float value : vector) {
      sum += (double) value * value;
    }
    return Math.sqrt(sum);
  }

  private static float cosine(final float[] a, final float[] b) {
    float dot = 0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  private static String preview(final float[] vector) {
    int n = Math.min(EMBED_PREVIEW, vector.length);
    StringBuilder text = new StringBuilder("[");
    for (int i = 0; i < n; i++) {
      if (i > 0) {
        text.append(", ");
      }
      text.append(String.format(Locale.ROOT, "%.4f", vector[i]));
    }
    if (vector.length > n) {
      text.append(", …");
    }
    return text.append(']').toString();
  }

  enum RagMode {
    NONE,
    BM25,
    DENSE,
    HYBRID
  }

  enum EncoderSession {
    EMBED,
    CLASSIFY
  }

  private enum ClassifyCommand {
    HELP,
    FORGET,
    LABELS,
    DEMO,
    LOAD;

    static Optional<ClassifyCommand> parse(final String user) {
      String command = user.toLowerCase(Locale.ROOT);
      if (command.equals("/help") || command.equals("help")) {
        return Optional.of(HELP);
      }
      if (command.equals("/forget") || command.equals("/clear")) {
        return Optional.of(FORGET);
      }
      if (command.equals("/labels")) {
        return Optional.of(LABELS);
      }
      if (command.equals("/demo")) {
        return Optional.of(DEMO);
      }
      if (command.equals("/load") || command.startsWith("/load ")) {
        return Optional.of(LOAD);
      }
      return Optional.empty();
    }
  }

  record ModelChoice(String label, Optional<Path> path, String missingHint, boolean missing) {
    String display() {
      return this.missing ? this.label + "  [not downloaded]" : this.label;
    }

    Path requirePath() {
      return this.path.orElseThrow(() -> new IllegalStateException(this.missingHint));
    }
  }

  private record AdvisorRole(String name, String prompt) {
    LlmAdvisor toAdvisor() {
      return LlmAdvisor.builder().name(this.name).prompt(this.prompt).build();
    }
  }
}
