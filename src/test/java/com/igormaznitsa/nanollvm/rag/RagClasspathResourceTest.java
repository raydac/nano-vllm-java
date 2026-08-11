package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

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
  void missingClasspathResourceFailsFast() {
    IllegalArgumentException ex = assertThrows(
      IllegalArgumentException.class,
      () -> RagFactory.makeResource(
        RagClasspathResourceTest.class.getClassLoader(),
        "rag-fixtures/does-not-exist.txt"));
    assertTrue(ex.getMessage().contains("not found"));
  }
}
