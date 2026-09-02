package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmLabelScore;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutLabels;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Minimal Meta fastText language-id demo (default {@code models/fasttext-lid-176}): load
 * {@code lid.176.bin} (or {@code lid.176.ftz} if that is all you have), classify sample
 * sentences, print top labels with scores.
 *
 * <p>This is a supervised fastText classifier — {@link LLM#builder} then
 * {@link LLM#generate} with {@link LlmInText} → {@link LlmModality#LABELS}. Do not use a
 * chat or embedding model here. Official models:
 * <a href="https://fasttext.cc/docs/en/language-identification.html">Language identification</a>.
 *
 * <p>Args: optional model folder or {@code *.ftz}/{@code *.bin} path (default
 * {@code models/fasttext-lid-176}), then optional text (default multi-language samples).
 * From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LanguageIdHelloWorld}
 *
 * @since 1.4.0
 */
public final class LanguageIdHelloWorld {

  private static final List<String> DEFAULT_SAMPLES = List.of(
    "The capital of France is Paris.",
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Привет, как дела?",
    "Hola, ¿cómo estás?",
    "これは日本語の文章です。"
  );

  private LanguageIdHelloWorld() {
  }

  public static void main(final String[] args) {
    Path path = modelPath(args);
    List<String> texts = seedTexts(args);

    System.out.println("Loading fastText language-id model from " + path);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model).build()) {
      classifyTexts(llm, texts);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static void classifyTexts(final LLM llm, final List<String> texts) {
    LlmModel model = llm.model();
    System.out.println("architecture=" + model.architectureName()
      + " classification=" + model.isClassificationModel()
      + " modalities=" + model.usableModalities());

    for (String text : texts) {
      LlmOutLabels out = (LlmOutLabels) llm.generate(LlmInText.of(text), LlmModality.LABELS);
      System.out.println("text: " + text);
      System.out.println("  top: " + formatLabel(out.top()));
      out.labels().stream()
        .skip(1)
        .limit(2)
        .forEach(score -> System.out.println("       " + formatLabel(score)));
    }
  }

  private static String formatLabel(final LlmLabelScore score) {
    return "%s  p=%.4f".formatted(stripLabelPrefix(score.label()), score.score());
  }

  private static String stripLabelPrefix(final String label) {
    return label.startsWith("__label__") ? label.substring("__label__".length()) : label;
  }

  private static Path modelPath(final String[] args) {
    return cliArg(args, 0)
      .map(path -> Path.of(path).toAbsolutePath().normalize())
      .orElseGet(BundledModels::requireFastTextLid);
  }

  private static List<String> seedTexts(final String[] args) {
    if (args == null || args.length < 2) {
      return DEFAULT_SAMPLES;
    }
    String joined = Arrays.stream(args, 1, args.length)
      .filter(part -> part != null && !part.isBlank())
      .collect(joining(" "));
    return joined.isBlank() ? DEFAULT_SAMPLES : List.of(joined);
  }

  private static Optional<String> cliArg(final String[] args, final int index) {
    if (args == null || args.length <= index || args[index] == null || args[index].isBlank()) {
      return Optional.empty();
    }
    return Optional.of(args[index]);
  }
}
