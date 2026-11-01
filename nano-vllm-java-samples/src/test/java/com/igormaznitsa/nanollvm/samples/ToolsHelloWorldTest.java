package com.igormaznitsa.nanollvm.samples;

import static java.nio.file.Files.createDirectory;
import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class ToolsHelloWorldTest {

  private static String jsonDirectory(final Path path) {
    return "{\"directory\":\"" + path.toString().replace("\\", "\\\\").replace("\"", "\\\"") +
      "\"}";
  }

  @Test
  void operatingSystemUsesTheRuntimeName() {
    assertTrue(ToolsHelloWorld.operatingSystem().startsWith(System.getProperty("os.name").strip()));
  }

  @Test
  void listFilesNamesTheDirectoryAndMarksFolders() throws IOException {
    Path root = createTempDirectory("os-files");
    writeString(root.resolve("beta.txt"), "x");
    createDirectory(root.resolve("alpha"));

    assertEquals("alpha/\nbeta.txt", ToolsHelloWorld.listFiles(jsonDirectory(root)));
  }

  @Test
  void listFilesRejectsAFileAndCapsALongDirectory() throws IOException {
    Path root = createTempDirectory("os-files-cap");
    Path file = root.resolve("one.txt");
    writeString(file, "x");
    assertTrue(ToolsHelloWorld.listFiles(jsonDirectory(file)).startsWith("error:"));

    IntStream.rangeClosed(0, ToolsHelloWorld.MAX_LISTED).forEach(index -> {
      try {
        writeString(root.resolve("f%03d.txt".formatted(index)), "x");
      } catch (IOException ex) {
        throw new IllegalStateException(ex);
      }
    });
    String listed = ToolsHelloWorld.listFiles(jsonDirectory(root));
    assertTrue(listed.contains("f000.txt"));
    assertTrue(listed.endsWith("… more"));
    assertFalse(listed.contains("f%03d.txt".formatted(ToolsHelloWorld.MAX_LISTED)));
  }

  @Test
  void omittedDirectoryIsTheWorkingDirectory() {
    assertEquals(
      Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize(),
      ToolsHelloWorld.resolveDirectory("{}"));
    assertEquals(
      Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize(),
      ToolsHelloWorld.resolveDirectory(""));
  }
}
