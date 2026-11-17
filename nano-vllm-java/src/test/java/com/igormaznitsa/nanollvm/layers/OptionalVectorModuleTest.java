package com.igormaznitsa.nanollvm.layers;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

final class OptionalVectorModuleTest {

  private static String codeLocation(final Class<?> type) throws Exception {
    URI location = type.getProtectionDomain().getCodeSource().getLocation().toURI();
    return Path.of(location).toString();
  }

  @Test
  void attentionAndDepthwiseConvRunWithoutTheVectorModule() throws Exception {
    Path java = Path.of(System.getProperty("java.home"), "bin", "java");
    Path javaExe = java.resolveSibling("java.exe");
    String command = Files.isExecutable(javaExe) ? javaExe.toString() : java.toString();
    String classpath = codeLocation(ScaledDotProductAttention.class)
      + File.pathSeparator
      + codeLocation(AttentionWithoutVectorProbe.class);
    ProcessBuilder builder = new ProcessBuilder(
      command,
      "-cp",
      classpath,
      AttentionWithoutVectorProbe.class.getName());
    builder.environment().remove("JAVA_TOOL_OPTIONS");
    builder.environment().remove("JDK_JAVA_OPTIONS");
    builder.environment().remove("_JAVA_OPTIONS");
    builder.redirectErrorStream(true);
    Process process = builder.start();
    boolean finished = process.waitFor(30, TimeUnit.SECONDS);
    if (!finished) {
      process.destroyForcibly();
    }
    String output;
    try (InputStream in = process.getInputStream()) {
      output = new String(in.readAllBytes(), UTF_8);
    }
    assertTrue(finished, output);
    assertFalse(output.contains("vector-visible"), output);
    assertEquals(0, process.exitValue(), output);
    assertTrue(output.contains("ok"), output);
  }
}
