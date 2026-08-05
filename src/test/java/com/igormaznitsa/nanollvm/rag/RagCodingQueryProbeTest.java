package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.utils.BundledRag;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class RagCodingQueryProbeTest {

  @Test
  void codingRequestAgainstBundledFairyTaleCorpusReturnsNoHits() {
    Optional<Path> root = BundledRag.find();
    if (root.isEmpty()) {
      return;
    }
    PreparedRag rag = RagFactory.make(root.get(), RagLoadOptions.forTinyModels());
    String coding =
        "write a Java program reading a file and printing its text lines one by one";
    assertTrue(rag.retrieve(coding, 4).isEmpty());
    assertTrue(rag.bm25().selectedQueryTerms(coding).isEmpty());
  }

  @Test
  void grimmNameQuestionsRetrieveAgainstBundledCorpus() {
    Optional<Path> root = BundledRag.find();
    if (root.isEmpty()) {
      return;
    }
    PreparedRag rag = RagFactory.make(root.get(), RagLoadOptions.forTinyModels());
    assertFalse(rag.retrieve("Grimm brothers", 2).isEmpty());
    assertFalse(rag.retrieve("names of the Grimm Brothers", 2).isEmpty());
    List<RagHit> hits = rag.retrieve(
        "who are the Grimm brothers and which their names are?", 4);
    assertFalse(hits.isEmpty());
    String joined =
        hits.stream().map(h -> h.chunk().text().toLowerCase()).reduce("", String::concat);
    assertTrue(joined.contains("jacob") || joined.contains("wilhelm") || joined.contains("grimm"));
  }
}
