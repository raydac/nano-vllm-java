package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;

import java.nio.file.Path;

/**
 * Minimal BERT embedding demo (GTE-small GGUF).
 *
 * <p>Args: optional {@code .gguf} path (default {@code models/gte-small.Q2_K.gguf}).
 * Example: {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.EmbedHelloWorld}
 *
 * @since 1.1.0
 */
public final class EmbedHelloWorld {

  private EmbedHelloWorld() {
  }

  public static void main(final String[] args) {
    Path path = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.GTE_SMALL_GGUF);

    System.out.println("Loading embedding model from " + path);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(path)) {
      System.out.println("architecture=" + model.architectureName()
        + " embedding=" + model.isEmbeddingModel());

      float[] hello = model.embed("hello world");
      float[] again = model.embed("hello world");
      float[] other = model.embed("a completely different sentence about astronomy");

      System.out.printf("dim=%d  L2(hello)=%.4f  cos(same)=%.4f  cos(diff)=%.4f%n",
        hello.length,
        l2(hello),
        cosine(hello, again),
        cosine(hello, other));
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static double l2(final float[] v) {
    double sum = 0.0;
    for (float x : v) {
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
}
