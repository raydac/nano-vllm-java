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
}
