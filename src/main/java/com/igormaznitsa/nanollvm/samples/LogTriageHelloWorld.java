package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Small Gemma3 demo: triage a payment-api log excerpt into three short bullets.
 *
 * <p>Args: optional model directory (default {@code models/Gemma3-270M} via
 * {@link BundledModels}). Example:
 * {@code mvn -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LogTriageHelloWorld
 * -Dexec.args=/opt/models/Gemma3-270M}
 */
public final class LogTriageHelloWorld {

  private LogTriageHelloWorld() {
  }

  public static void main(final String[] args) {
    Path gemmaDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.GEMMA3_270M);

    System.out.println("Loading Gemma3 from " + gemmaDir);
    long totalStarted = System.nanoTime();

    long loadStarted = System.nanoTime();
    try (LlmModel model = LlmModelFactory.make(gemmaDir)) {
      printTiming("model load", System.nanoTime() - loadStarted);

      String prompt = """
        You are helping an on-call engineer. Read the log lines and reply in three short bullets:
        1) what failed
        2) likely cause
        3) one concrete next check
        Do not invent hosts or error codes that are not in the log.

        Log:
        2026-08-07 22:14:01 WARN  payment-api - retry 1/3 for order=99102 cause=SocketTimeoutException
        2026-08-07 22:14:04 WARN  payment-api - retry 2/3 for order=99102 cause=SocketTimeoutException
        2026-08-07 22:14:08 ERROR payment-api - give up order=99102 after 3 timeouts upstream=billing-svc:8443
        2026-08-07 22:14:08 INFO  payment-api - marked order=99102 status=PAYMENT_FAILED
        """;

      long buildStarted = System.nanoTime();
      try (LLM llm = LLM.builder(model)
        .noSystemPrompt()
        .disableMultiCpu()
        .maxModelLen(2048)
        .listen(LlmListeners.toSystem())
        .build()) {
        printTiming("engine build", System.nanoTime() - buildStarted);

        long chatStarted = System.nanoTime();
        String advice = llm.chat(128).send(prompt).answer();
        printTiming("chat turn", System.nanoTime() - chatStarted);

        System.out.println();
        System.out.println(advice);
      }
    }

    printTiming("total", System.nanoTime() - totalStarted);
  }

  private static void printTiming(final String action, final long nanos) {
    double millis = nanos / 1_000_000.0;
    String duration = millis < 1_000.0
      ? "%.0f ms".formatted(millis)
      : "%.2f s".formatted(millis / 1_000.0);
    System.out.printf(Locale.ROOT, "[timing] %s: %s%n", action, duration);
  }
}
