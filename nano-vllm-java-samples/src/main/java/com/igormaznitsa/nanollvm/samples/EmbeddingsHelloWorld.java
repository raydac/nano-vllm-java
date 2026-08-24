package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Minimal sentence-embedding demo (default multilingual-e5-small ONNX): encode text to an
 * L2-normalized vector, print dim / preview, then cosine vs the same text and an unrelated
 * sentence.
 *
 * <p>This is a BERT embedding checkpoint — {@link com.igormaznitsa.nanollvm.llm.LLM#builder} then
 * {@link com.igormaznitsa.nanollvm.llm.LLM#embed}. Do not use a chat/completion model here. Non-retrieval E5 inputs use the
 * {@code query: } prefix.
 *
 * <p>Args: optional model folder (default {@code models/multilingual-e5-small}), optional text
 * (default {@code The capital of France is Paris.}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.EmbeddingsHelloWorld}
 *
 * @since 1.1.0
 */
public final class EmbeddingsHelloWorld {

  private static final int PREVIEW = 8;
  private static final String QUERY_PREFIX = "query: ";
  private static final String DEFAULT_TEXT = "The capital of France is Paris.";
  private static final String RELATED_TEXT = "Paris is the capital city of France.";
  private static final String UNRELATED_TEXT = "A completely different sentence about astronomy.";

  private EmbeddingsHelloWorld() {
  }

  public static void main(final String[] args) {
    Path path = modelPath(args);
    String text = seedText(args);

    System.out.println("Loading embedding model from " + path);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model).build()) {
      embedTexts(llm, text, usesE5Prefix(path));
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static void embedTexts(
    final LLM llm,
    final String text,
    final boolean e5Prefix
  ) {
    LlmModel model = llm.model();
    System.out.println("architecture=" + model.architectureName()
      + " embedding=" + model.isEmbeddingModel()
      + " modalities=" + model.modalities());

    float[] vector = llm.embed(embedInput(text, e5Prefix));
    float[] same = llm.embed(embedInput(text, e5Prefix));
    float[] other = llm.embed(embedInput(UNRELATED_TEXT, e5Prefix));

    printVector(text, vector);
    System.out.printf(Locale.ROOT, "cos(same)=%.4f%n", cosine(vector, same));

    if (DEFAULT_TEXT.equals(text)) {
      float[] related = llm.embed(embedInput(RELATED_TEXT, e5Prefix));
      System.out.printf(Locale.ROOT, "cos(related)=%.4f  %s%n", cosine(vector, related),
        RELATED_TEXT);
    }

    System.out.printf(Locale.ROOT, "cos(unrelated)=%.4f  %s%n", cosine(vector, other),
      UNRELATED_TEXT);
  }

  private static void printVector(final String text, final float[] vector) {
    System.out.println("text: " + text);
    System.out.printf(Locale.ROOT, "dim=%d  L2=%.4f%n", vector.length, l2(vector));
    System.out.println("preview: " + preview(vector));
  }

  private static String preview(final float[] vector) {
    int n = Math.min(PREVIEW, vector.length);
    String values = IntStream.range(0, n)
      .mapToObj(i -> String.format(Locale.ROOT, "%.4f", vector[i]))
      .collect(joining(", "));
    return vector.length > n ? "[" + values + ", …]" : "[" + values + "]";
  }

  private static double l2(final float[] vector) {
    double sum = 0.0;
    for (float x : vector) {
      sum += (double) x * x;
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

  private static Path modelPath(final String[] args) {
    return cliArg(args, 0)
      .map(path -> Path.of(path).toAbsolutePath().normalize())
      .orElseGet(() -> BundledModels.require(BundledModels.MULTILINGUAL_E5_SMALL));
  }

  private static boolean usesE5Prefix(final Path path) {
    return path.getFileName().toString().toLowerCase(Locale.ROOT).contains("e5");
  }

  private static String embedInput(final String text, final boolean e5Prefix) {
    return e5Prefix ? QUERY_PREFIX + text : text;
  }

  private static String seedText(final String[] args) {
    if (args == null || args.length < 2) {
      return DEFAULT_TEXT;
    }
    String joined = Arrays.stream(args, 1, args.length)
      .filter(part -> part != null && !part.isBlank())
      .collect(joining(" "));
    return joined.isBlank() ? DEFAULT_TEXT : joined;
  }

  private static Optional<String> cliArg(final String[] args, final int index) {
    if (args == null || args.length <= index || args[index] == null || args[index].isBlank()) {
      return Optional.empty();
    }
    return Optional.of(args[index]);
  }
}
