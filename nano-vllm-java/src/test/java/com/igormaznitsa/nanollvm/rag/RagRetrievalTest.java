package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RagRetrievalTest {

  @Test
  void shortFollowUpExpandsWithAnchorQuery() {
    assertTrue(RagSession.Retrieval.shortFollowUp("what are their names?"));
    assertFalse(RagSession.Retrieval.shortFollowUp(
        "What are the full given names of both Grimm brothers in the biography?"));
    assertEquals(
        "names of the Grimm Brothers are\nwhat are their names?",
      RagSession.Retrieval.anchorExpandedQuery(
        "what are their names?", "names of the Grimm Brothers are"));
    assertTrue(RagSession.Retrieval.updatesAnchorFromQuestion("names of the Grimm Brothers are"));
    assertFalse(RagSession.Retrieval.updatesAnchorFromQuestion("what are their names?"));
  }

  @Test
  void preferPriorSourceKeepsCompetitiveSameFile() {
    TextChunk hood = new TextChunk("h", "/rag/hood.txt", "Little Red Riding Hood met a wolf.");
    TextChunk grimm =
        new TextChunk("g", "/rag/grimm.txt", "Jacob and Wilhelm Grimm were brothers.");
    List<RagHit> candidates = List.of(
        new RagHit(hood, 1.0),
        new RagHit(grimm, 0.7));
    List<RagHit> preferred =
      RagSession.Retrieval.preferPriorSource(candidates, "/rag/grimm.txt", 1);
    assertEquals(1, preferred.size());
    assertTrue(preferred.getFirst().chunk().text().contains("Jacob"));
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
  void offTopicProperNameDoesNotRetrieveFairyTaleNoise() {
    PreparedRag index = RagFactory.of(
      "Little Red Riding Hood met a wolf. Oh grandmother, what big ears you have! "
        + "She did not know what a wicked animal he was, and was not afraid of him. "
        + "I usually like it at grandmother's.");
    assertTrue(index.isOutsideCorpus("what do you think about estonia?"));
    assertTrue(index.retrieve("what do you think about estonia?", 3).isEmpty());
    assertTrue(index.isOutsideCorpus("I would like to discuss about cars"));
    assertTrue(index.retrieve("I would like to discuss about cars", 3).isEmpty());
    assertTrue(index.isOutsideCorpus("what do you think about BMW?"));
    assertTrue(index.retrieve("what do you think about BMW?", 3).isEmpty());
    assertFalse(index.retrieve("Little Red Riding Hood wolf grandmother", 2).isEmpty());
  }

  @Test
  void conversationalGlueDoesNotKeepOffTopicBmwInsideCorpus() {
    PreparedRag index = RagFactory.of(
      "Little Red Riding Hood met a wolf. She thought about the path. "
        + "What do you think the grandmother will say? I think about flowers.");
    assertTrue(index.isOutsideCorpus("what do you think about BMW?"));
    assertTrue(index.retrieve("what do you think about BMW?", 3).isEmpty());
    assertFalse(index.isOutsideCorpus("Little Red Riding Hood wolf grandmother"));
  }

  @Test
  void offTopicCodingRequestDoesNotRetrieveOnWeakCorpusOverlap() {
    PreparedRag index = RagFactory.of(
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
    assertTrue(RagSession.Retrieval.shortFollowUp("write a Java program without explanation"));
    assertEquals(
        "names of the Grimm Brothers\nwrite a Java program without explanation",
      RagSession.Retrieval.anchorExpandedQuery(
            "write a Java program without explanation",
            "names of the Grimm Brothers"));

    PreparedRag rag = RagFactory.of(
        RagLoadOptions.defaults(),
        "Jacob Grimm and Wilhelm Grimm were the Brothers Grimm. Their names are Jacob and Wilhelm.");
    assertTrue(rag.isOutsideCorpus("write a Java program without explanation"));
    assertFalse(rag.isOutsideCorpus("what are their names?"));
    assertFalse(rag.retrieve("Jacob Wilhelm Grimm brothers names", 2).isEmpty());
  }

  @Test
  void rewrittenGrimmKeywordsPreferGrimmSourceOverHoodKnowWhatNoise() {
    PreparedRag rag = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .add("grimm.txt",
        "The Brothers Grimm were Jacob Grimm and Wilhelm Grimm. Their names were Jacob and Wilhelm.")
      .add("hood.txt",
        "She did not know what a wicked animal he was, and was not afraid of him. "
          + "You must know the place, said Little Red Riding Hood. "
          + "Oh grandmother, what big ears you have!")
      .build();

    List<RagHit> hits = rag.retrieve("Grimm brothers Jacob Wilhelm", 2);

    assertFalse(hits.isEmpty());
    assertTrue(
      hits.getFirst().chunk().source().contains("grimm"),
      () -> "expected Grimm source, got " + hits.stream().map(h -> h.chunk().source()).toList());
  }

  @Test
  void shortFollowUpMatchesShortQuestionsOnly() {
    assertTrue(RagSession.Retrieval.shortFollowUp("name of their father"));
    assertFalse(RagSession.Retrieval.shortFollowUp(
      "What are the full given names of both Grimm brothers in the biography?"));
  }

  @Test
  void longNaturalGrimmQuestionHitsTinyModelCorpus() {
    Path ragDir = OptionalModelAssumptions.requireLocalRag();
    PreparedRag rag = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .addFolder(ragDir)
      .build();

    String q = "who are the grimm brothers and what did they do?";
    assertFalse(rag.isOutsideCorpus(q));
    List<RagHit> hits = rag.retrieve(q, 3);
    assertFalse(hits.isEmpty(), "expected Grimm hits for natural long question");
    assertTrue(
      hits.stream().anyMatch(h -> h.chunk().text().toLowerCase(Locale.ROOT).contains("grimm")),
      () -> "expected Grimm passage, got " + hits.stream().map(h -> h.chunk().text()).toList());
  }

  @Test
  void fatherQueriesHitBundledCorpusWithoutRewrite() {
    PreparedRag rag = RagFactory.make(OptionalModelAssumptions.requireLocalRag());

    assertTrue(RagSession.Retrieval.shortFollowUp("who was their father?"));
    assertTrue(RagSession.Retrieval.shortFollowUp("father of grimm brothers"));
    assertTrue(RagSession.Retrieval.hasHits(rag, "who was their father?"));
    assertTrue(RagSession.Retrieval.hasHits(rag, "father of grimm brothers"));

    List<RagHit> fatherHits = rag.retrieve("father of grimm brothers", 2);
    assertFalse(fatherHits.isEmpty());
    assertTrue(
      fatherHits.getFirst().chunk().text().toLowerCase(Locale.ROOT).contains("father"),
      () -> "expected father passage, got: " + fatherHits.getFirst().chunk().text());
  }

  @Test
  void rewriteNoneFallsBackToAnchorExpansion() {
    PreparedRag rag = RagFactory.make(OptionalModelAssumptions.requireLocalRag());
    Optional<String> chosen = RagSession.Retrieval.queryAfterRewrite(
      "who was their father?",
      "who are the grimm brothers?",
      null,
      rag);
    assertEquals(
      "who are the grimm brothers?\nwho was their father?",
      chosen.orElseThrow());
    assertFalse(rag.retrieve(chosen.orElseThrow(), 2).isEmpty());
  }

  @Test
  void rewriteOutsideCorpusFallsBackToAnchorExpansion() {
    PreparedRag rag = RagFactory.make(OptionalModelAssumptions.requireLocalRag());
    Optional<String> chosen = RagSession.Retrieval.queryAfterRewrite(
      "who was their father?",
      "who are the grimm brothers?",
      "estonia capital tallinn",
      rag);
    assertEquals(
      "who are the grimm brothers?\nwho was their father?",
      chosen.orElseThrow());
  }


  @Test
  void shortAnaphoricFollowUpsStillNeedAnchor() {
    assertTrue(RagSession.Retrieval.shortFollowUp("their names?"));
    assertTrue(RagSession.Retrieval.shortFollowUp("names of the Grimm brothers"));
  }

  @Test
  void cyrillicInflectionKeysMatchCaseVariantsWithoutExternalCorpus() {
    PreparedRag rag = RagFactory.of(
      RagLoadOptions.defaults(),
      "Сказка о колобке и о бабе яге из русских народных сказок.");

    assertFalse(rag.retrieve("колобок", 2).isEmpty());
    assertFalse(rag.retrieve("Баба Яга", 2).isEmpty());
  }

  @Test
  void tryMakeReturnsEmptyForEmptyFolder(@TempDir final Path tempDir) {
    assertTrue(RagFactory.tryMake(tempDir).isEmpty());
  }

  @Test
  void tryMakeReturnsEmptyWhenOnlyReadmePresent(@TempDir final Path tempDir) throws Exception {
    Files.writeString(tempDir.resolve("README.md"), "# notes\n", UTF_8);
    assertTrue(RagFactory.tryMake(tempDir).isEmpty());
  }

  @Test
  void ragLoadReportsPerFileExtractionStats(@TempDir final Path tempDir) throws Exception {
    Path note = tempDir.resolve("notes.txt");
    Files.writeString(note, "Paris is the capital of France.", UTF_8);

    ByteArrayOutputStream err = new ByteArrayOutputStream();
    PrintStream sink = new PrintStream(err, true, UTF_8);
    LlmListener io = LlmListeners.ofStatusStreams(sink, sink);

    PreparedRag prepared = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .listen(io)
      .addFile(note)
      .build();

    String log = err.toString(UTF_8);
    assertTrue(log.contains("notes.txt"));
    assertTrue(log.contains("char(s)"));
    assertTrue(log.contains("chunk(s)"));
    assertTrue(log.contains("RAG ready"));
    assertFalse(prepared.retrieve("Paris France capital", 1).isEmpty());
  }
}
