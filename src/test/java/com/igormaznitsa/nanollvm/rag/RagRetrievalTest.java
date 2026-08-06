package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class RagRetrievalTest {

  @Test
  void shortFollowUpExpandsWithAnchorQuery() {
    assertTrue(RagRetrieval.needsAnchor("what are their names?"));
    assertFalse(RagRetrieval.needsAnchor(
        "What are the full given names of both Grimm brothers in the biography?"));
    assertEquals(
        "names of the Grimm Brothers are\nwhat are their names?",
        RagRetrieval.retrievalQuery("what are their names?", "names of the Grimm Brothers are"));
    assertTrue(RagRetrieval.shouldUpdateAnchor("names of the Grimm Brothers are"));
    assertFalse(RagRetrieval.shouldUpdateAnchor("what are their names?"));
  }

  @Test
  void preferPriorSourceKeepsCompetitiveSameFile() {
    TextChunk hood = new TextChunk("h", "/rag/hood.txt", "Little Red Riding Hood met a wolf.");
    TextChunk grimm =
        new TextChunk("g", "/rag/grimm.txt", "Jacob and Wilhelm Grimm were brothers.");
    List<RagHit> candidates = List.of(
        new RagHit(hood, 1.0),
        new RagHit(grimm, 0.7));
    List<RagHit> preferred = RagRetrieval.preferPriorSource(candidates, "/rag/grimm.txt", 1);
    assertEquals(1, preferred.size());
    assertTrue(preferred.getFirst().chunk().text().contains("Jacob"));
  }

  @Test
  void preferCompactPassagesKeepsShorterCompetitiveHit() {
    TextChunk longPass = new TextChunk(
      "long",
      "/docs/chapter.txt",
      "A".repeat(400) + " capital city of France is discussed across many pages of history.");
    TextChunk shortPass = new TextChunk(
      "short",
      "/docs/notes.txt",
      "Paris is the capital of France.");
    List<RagHit> candidates = List.of(
      new RagHit(longPass, 1.0),
      new RagHit(shortPass, 0.7));
    List<RagHit> preferred = RagRetrieval.preferCompactPassages(candidates, 2);
    assertEquals(1, preferred.size());
    assertTrue(preferred.getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void preparedRagPrefersShorterDensePassage() {
    PreparedRag prepared = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .add("chapter.txt",
        "The capital of France appears in a long travelogue about rivers, kings, and museums.")
      .add("notes.txt", "Paris is the capital of France.")
      .build();
    List<RagHit> hits = prepared.retrieve("capital of France Paris", 1);
    assertFalse(hits.isEmpty());
    assertTrue(hits.getFirst().chunk().text().contains("Paris is the capital"));
  }

  @Test
  void offTopicCodingRequestDoesNotRetrieveOnWeakCorpusOverlap() {
    Bm25Index index = Bm25Index.of(
        "In 1829 the position should have been awarded by Jacob Grimm, "
            + "but another person, one without any merit, was preferred. "
            + "See also the text about their work.");
    assertTrue(index.retrieve(
        "write a Java program reading a file and printing its text lines one by one",
        2).isEmpty());
    assertTrue(index.retrieve("please write any Java program", 2).isEmpty());
    assertFalse(index.retrieve("Jacob Grimm awarded position", 1).isEmpty());
  }

  @Test
  void shortCodingFollowUpDoesNotInheritGrimmAnchor() {
    assertTrue(RagRetrieval.needsAnchor("write a Java program without explanation"));
    assertEquals(
        "names of the Grimm Brothers\nwrite a Java program without explanation",
        RagRetrieval.retrievalQuery(
            "write a Java program without explanation",
            "names of the Grimm Brothers"));

    PreparedRag rag = RagFactory.of(
        RagLoadOptions.defaults(),
        "Jacob Grimm and Wilhelm Grimm were the Brothers Grimm. Their names are Jacob and Wilhelm.");
    assertTrue(rag.bm25().isOutsideCorpus("write a Java program without explanation"));
    assertFalse(rag.bm25().isOutsideCorpus("what are their names?"));
    assertFalse(rag.retrieve("Jacob Wilhelm Grimm brothers names", 2).isEmpty());
  }
}
