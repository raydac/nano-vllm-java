package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.createTempFile;
import static java.nio.file.Files.deleteIfExists;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Load-time {@link RagTuner} filter, extract, and preprocess chain.
 */
class RagTunerTest {

  @Test
  void filterSkipsDisallowedFilesInFolder() throws Exception {
    Path dir = createTempDirectory("nanollvm-rag-tuner-filter");
    Path keep = dir.resolve("keep.txt");
    Path skip = dir.resolve("skip.txt");
    writeString(keep, "Paris is the capital of France.");
    writeString(skip, "Berlin is a city in Germany.");
    try {
      PreparedRag rag = RagFactory.builder()
        .addProcessor(RagTuner.allowing(resource -> !resource.fileName().equals("skip.txt")))
        .addFolder(dir)
        .build();

      assertTrue(
        rag.retrieve("Paris France capital", 1).getFirst().chunk().text().contains("Paris"));
      assertTrue(rag.retrieve("Berlin Germany city", 1).isEmpty());
    } finally {
      deleteIfExists(keep);
      deleteIfExists(skip);
      deleteIfExists(dir);
    }
  }

  @Test
  void anyRejectingFilterSkipsTheResource() throws Exception {
    Path file = createTempFile("nanollvm-rag-tuner-reject", ".txt");
    writeString(file, "Paris is the capital of France.");
    try {
      assertThrows(IllegalStateException.class, () -> RagFactory.builder()
        .addProcessor(
          RagTuner.allowing(resource -> true),
          RagTuner.allowing(resource -> false))
        .addFile(file)
        .build());
    } finally {
      deleteIfExists(file);
    }
  }

  @Test
  void filterDoesNotDropInlineText() {
    PreparedRag rag = RagFactory.builder()
      .addProcessor(RagTuner.allowing(resource -> false))
      .add("Paris is the capital of France.")
      .build();

    assertTrue(rag.retrieve("Paris France capital", 1).getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void customExtractReplacesFileBytes() throws Exception {
    Path file = createTempFile("nanollvm-rag-tuner-extract", ".txt");
    writeString(file, "IGNORE THIS BODY");
    try {
      PreparedRag rag = RagFactory.builder()
        .addProcessor(RagTuner.extracting(resource -> Optional.of("Cats are mammals.")))
        .addFile(file)
        .build();

      assertTrue(rag.retrieve("mammals cats", 1).getFirst().chunk().text().contains("Cats"));
    } finally {
      deleteIfExists(file);
    }
  }

  @Test
  void emptyExtractFallsBackToStandardLoader() throws Exception {
    Path file = createTempFile("nanollvm-rag-tuner-fallback", ".txt");
    writeString(file, "Tokyo is the capital of Japan.");
    try {
      PreparedRag rag = RagFactory.builder()
        .addProcessor(RagTuner.extracting(resource -> Optional.empty()))
        .addFile(file)
        .build();

      assertTrue(
        rag.retrieve("Tokyo Japan capital", 1).getFirst().chunk().text().contains("Tokyo"));
    } finally {
      deleteIfExists(file);
    }
  }

  @Test
  void firstPresentExtractWins() throws Exception {
    Path file = createTempFile("nanollvm-rag-tuner-first", ".txt");
    writeString(file, "original text is ignored");
    try {
      PreparedRag rag = RagFactory.builder()
        .addProcessor(
          RagTuner.extracting(resource -> Optional.of("Cats are mammals.")),
          RagTuner.extracting(resource -> Optional.of("Dogs are mammals.")))
        .addFile(file)
        .build();

      String text = rag.retrieve("mammals", 1).getFirst().chunk().text();
      assertTrue(text.contains("Cats"));
      assertFalse(text.contains("Dogs"));
    } finally {
      deleteIfExists(file);
    }
  }

  @Test
  void preprocessChainAppliesInRegistrationOrder() {
    PreparedRag rag = RagFactory.builder()
      .addProcessor(
        RagTuner.preprocessing(text -> text.replace("X", "Paris")),
        RagTuner.preprocessing(text -> text + " is the capital of France."))
      .add("X")
      .build();

    assertTrue(
      rag.retrieve("Paris France capital", 1).getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void classpathExtractAndFilter() {
    PreparedRag extracted = RagFactory.builder()
      .addProcessor(RagTuner.extracting(resource -> {
        assertTrue(resource.isClasspath());
        assertTrue(resource.hasContent());
        return Optional.of("Mercury is the closest planet to the Sun.");
      }))
      .addResource(RagTunerTest.class, "/rag-fixtures/paris.txt")
      .build();
    assertTrue(
      extracted.retrieve("Mercury planet Sun", 1).getFirst().chunk().text().contains("Mercury"));

    assertThrows(IllegalStateException.class, () -> RagFactory.builder()
      .addProcessor(RagTuner.allowing(resource -> !resource.fileName().equals("paris.txt")))
      .addResource(RagTunerTest.class.getClassLoader(), "rag-fixtures/paris.txt")
      .build());
  }

  @Test
  void resourceFactoriesExposePathAndBytes() {
    Path path = Path.of("docs", "note.md");
    RagResource file = RagResource.file(path, "hello".getBytes(UTF_8));
    assertTrue(file.isFile());
    assertEquals("note.md", file.fileName());
    assertTrue(file.path().isPresent());
    assertEquals("hello", new String(file.content().orElseThrow(), UTF_8));

    RagResource classpath = RagResource.classpath("/rag/facts.md");
    assertTrue(classpath.isClasspath());
    assertEquals("facts.md", classpath.fileName());
    assertEquals("classpath:rag/facts.md", classpath.source());
    assertTrue(classpath.content().isEmpty());
  }
}
