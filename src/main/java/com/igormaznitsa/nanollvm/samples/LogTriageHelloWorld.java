package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;

import java.nio.file.Path;

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
    Path gemmaDir = resolveGemmaDir(args);
    LlmModel model = LlmModelFactory.make(gemmaDir);

    String logExcerpt = """
      2026-08-07 22:14:01 WARN  payment-api - retry 1/3 for order=99102 cause=SocketTimeoutException
      2026-08-07 22:14:04 WARN  payment-api - retry 2/3 for order=99102 cause=SocketTimeoutException
      2026-08-07 22:14:08 ERROR payment-api - give up order=99102 after 3 timeouts upstream=billing-svc:8443
      2026-08-07 22:14:08 INFO  payment-api - marked order=99102 status=PAYMENT_FAILED
      """;

    String prompt = """
      You are helping an on-call engineer. Read the log lines and reply in three short bullets:
      1) what failed
      2) likely cause
      3) one concrete next check
      Do not invent hosts or error codes that are not in the log.

      Log:
      %s
      """.formatted(logExcerpt);

    System.out.println("Loading Gemma3 from " + gemmaDir);
    try (LLM llm = LLM.builder(model)
      .noSystemPrompt()
      .skipWarmup()
      .disableMultiCpu()
      .maxModelLen(2048)
      .listen(LlmListeners.toSystem())
      .build()) {
      System.out.println(llm.chat(128).send(prompt).answer());
    }
  }

  private static Path resolveGemmaDir(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return Path.of(args[0]).toAbsolutePath().normalize();
    }
    return BundledModels.require(BundledModels.GEMMA3_270M);
  }
}
