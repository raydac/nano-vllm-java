package com.igormaznitsa.nanollvm.rag;

import static java.nio.file.Files.createDirectory;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RagClasspathResourceTest {

  @Test
  void makeResourceLoadsAbsoluteClasspathDocument() {
    PreparedRag rag = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .addResource(RagClasspathResourceTest.class, "/rag-fixtures/paris.txt")
      .build();

    assertTrue(rag.size() >= 1);
    assertTrue(
      rag.chunks().stream().anyMatch(chunk -> chunk.source().startsWith("classpath:")),
      () -> "expected classpath source, got "
        + rag.chunks().stream().map(TextChunk::source).toList());

    List<RagHit> hits = rag.retrieve("capital of France Paris", 1);
    assertFalse(hits.isEmpty());
    assertTrue(hits.getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void addResourceWithClassLoaderFindsFixture() {
    PreparedRag rag = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .addResource(
        RagClasspathResourceTest.class.getClassLoader(),
        "rag-fixtures/paris.txt")
      .build();

    assertFalse(rag.retrieve("Paris France capital", 1).isEmpty());
  }

  @Test
  void builderMixesFoldersClasspathAndInline(final @TempDir Path root) throws Exception {
    Path kb = createDirectory(root.resolve("kb"));
    Path legal = createDirectory(root.resolve("legal"));
    writeString(kb.resolve("mars.txt"), "Mars is the fourth planet from the Sun.");
    writeString(legal.resolve("warranty.md"), "Warranty lasts two years from purchase.");

    PreparedRag rag = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .addFolders(kb, legal)
      .addResource(RagClasspathResourceTest.class, "/rag-fixtures/paris.txt")
      .add("support-hours", "Live chat is available 9 to 17 UTC.")
      .build();

    assertTrue(rag.retrieve("fourth planet Mars", 1).getFirst().chunk().text().contains("Mars"));
    assertTrue(
      rag.retrieve("Warranty two years", 1).getFirst().chunk().text().contains("Warranty"));
    assertTrue(
      rag.retrieve("capital of France Paris", 1).getFirst().chunk().text().contains("Paris"));
    assertTrue(rag.retrieve("Live chat UTC", 1).getFirst().chunk().text().contains("Live chat"));

    List<String> sources = rag.chunks().stream().map(TextChunk::source).toList();
    assertTrue(sources.stream().anyMatch(source -> source.startsWith("classpath:")));
    assertTrue(sources.stream().anyMatch(source -> source.contains("mars.txt")));
    assertTrue(sources.stream().anyMatch(source -> source.contains("warranty.md")));
    assertTrue(sources.contains("support-hours"));
  }

  @Test
  void missingClasspathResourceFailsFast() {
    IllegalArgumentException ex = assertThrows(
      IllegalArgumentException.class,
      () -> RagFactory.makeResource(
        RagClasspathResourceTest.class.getClassLoader(),
        "rag-fixtures/does-not-exist.txt"));
    assertTrue(ex.getMessage().contains("not found"));
  }
}
