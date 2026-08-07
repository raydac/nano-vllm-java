package com.igormaznitsa.nanollvm;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.createTempFile;
import static java.nio.file.Files.deleteIfExists;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class RagUnitTest {

  @Test
  void bm25PrefersMatchingString() {
    PreparedRag index = RagFactory.of(
        "Paris is the capital of France.",
        "Berlin is a city in Germany.",
        "Tokyo is the capital of Japan.");
    List<RagHit> hits = index.retrieve("capital of France", 2);
    assertFalse(hits.isEmpty());
    assertTrue(hits.getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void corpusFromFileAndFolder() throws Exception {
    Path dir = createTempDirectory("nanollvm-rag");
    Path a = dir.resolve("a.txt");
    Path b = dir.resolve("b.md");
    writeString(a, "The Nile is a long river in Africa.");
    writeString(b, "Mount Everest is the highest mountain.");
    try {
      PreparedRag fromFile = RagFactory.builder().addFile(a).build();
      assertEquals(1, fromFile.size());
      assertTrue(fromFile.retrieve("Nile river", 1).getFirst().chunk().text().contains("Nile"));

      PreparedRag fromFolder = RagFactory.make(dir);
      assertTrue(fromFolder.size() >= 2);
      assertTrue(fromFolder.retrieve("highest mountain", 1).getFirst().chunk().text()
          .contains("Everest"));
    } finally {
      deleteIfExists(a);
      deleteIfExists(b);
      deleteIfExists(dir);
    }
  }

  @Test
  void builderMixesStringsFilesAndFolder() throws Exception {
    Path file = createTempFile("nanollvm-rag", ".txt");
    Path dir = createTempDirectory("nanollvm-rag-docs");
    Path nested = dir.resolve("note.txt");
    writeString(file, "Mercury is the closest planet to the Sun.");
    writeString(nested, "Venus is the second planet from the Sun.");
    try {
      PreparedRag index = RagFactory.builder()
          .add("id-earth", "Earth is the third planet from the Sun.")
          .addFile(file)
          .addFolder(dir)
          .build();
      assertTrue(index.size() >= 3);
      assertTrue(index.retrieve("third planet", 1).getFirst().chunk().text().contains("Earth"));
    } finally {
      deleteIfExists(nested);
      deleteIfExists(dir);
      deleteIfExists(file);
    }
  }

  @Test
  void ragPromptIncludesContextAndQuestion() {
    PreparedRag index = RagFactory.of("Cats are mammals.");
    List<RagHit> hits = index.retrieve("mammals", 1);
    String prompt = RagSession.formatUserMessage(hits, "What are cats?");
    assertTrue(prompt.contains("Cats are mammals."));
    assertTrue(prompt.contains("What are cats?"));
    assertTrue(prompt.contains("Context:"));
  }

  @Test
  void compactPromptPutsQuestionBeforePassages() {
    PreparedRag index = RagFactory.of("The Nile is a major river in Africa.");
    List<RagHit> hits = index.retrieve("Nile", 1);
    String prompt = RagSession.formatUserMessage(hits, "what is Nile?", 900, true);
    assertTrue(prompt.contains("The Nile is a major river in Africa."));
    assertTrue(prompt.contains("what is Nile?"));
    assertTrue(prompt.indexOf("what is Nile?") < prompt.indexOf("The Nile is a major river"));
    assertTrue(prompt.contains("Context:"));
    assertTrue(prompt.contains("Answer in one short sentence"));
    assertFalse(prompt.contains("say you do not know"));
    assertFalse(prompt.contains("answer the request normally"));
    assertFalse(prompt.contains("previous answer was wrong"));
  }

  @Test
  void compactNoHitPromptForbidsInvention() {
    String prompt = RagSession.formatUserMessage(List.of(), "где живут ведьмы?", 900, true);
    assertTrue(prompt.contains("где живут ведьмы?"));
    assertTrue(prompt.contains("No context documents were found"));
    assertTrue(prompt.contains("I do not know"));
    assertTrue(prompt.contains("Do not invent"));
  }

  @Test
  void groundedPromptForbidsInventingDetails() {
    PreparedRag index = RagFactory.of("Jacob Grimm was born in 1785.");
    List<RagHit> hits = index.retrieve("Jacob Grimm born", 1);
    String prompt = RagSession.formatUserMessage(hits, "When was Jacob born?");
    assertTrue(prompt.contains("Do not invent"));
    assertTrue(prompt.contains("say you do not know"));
  }

  @Test
  void relativeScoreFilterDropsWeakSecondHit() {
    PreparedRag index = RagFactory.of(
        "Paris is the capital of France. France is a country in Europe.",
        "No CUDA kernels exist in this port of the engine.");
    List<RagHit> france = index.retrieve("france capital paris", 3);
    assertFalse(france.isEmpty());
    assertTrue(france.getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void preparedRagFromFactoryIsShareableAndRetrieves() {
    PreparedRag prepared = RagFactory.of(
        RagLoadOptions.defaults(),
        "Paris is the capital of France.",
        "Berlin is a city in Germany.");
    assertEquals(2, prepared.size());
    assertTrue(
        prepared.retrieve("capital of France", 1).getFirst().chunk().text().contains("Paris"));
    assertTrue(prepared.sourceRoot().isEmpty());
  }

  @Test
  void preprocessorKeepsSectionTitleOnSentences() {
    PreparedRag prepared = RagFactory.builder()
        .options(RagLoadOptions.forTinyModels())
        .add("""
            # Capitals

            Paris is the capital of France. Berlin is a city in Germany.
            """)
        .build();
    assertFalse(prepared.chunks().isEmpty());
    String joined = prepared.chunks().stream()
        .map(TextChunk::text)
        .collect(java.util.stream.Collectors.joining(" "));
    assertTrue(joined.contains("Capitals —"));
    assertTrue(joined.contains("Paris is the capital of France."));
    assertTrue(prepared.size() >= 2);
  }

  @Test
  void atomicLoadDedupesIdenticalPassages() {
    PreparedRag prepared = RagFactory.of(
        RagLoadOptions.forTinyModels(),
        "Paris is the capital of France.",
        "Paris is the capital of France.");
    assertEquals(1, prepared.size());
  }

  @Test
  void preparserInjectsSourceStemIntoSearchText() {
    PreparedRag prepared = RagFactory.builder()
      .add("facts-capitals.md", "Paris is the capital of France.")
      .build();
    PreparedRag.Passage passage = prepared.passages().getFirst();
    assertTrue(passage.searchText().contains("capitals"));
    assertTrue(passage.searchText().contains("facts"));
    assertTrue(passage.termFreqs().containsKey("paris"));
    assertTrue(passage.termFreqs().containsKey("capitals"));
    assertEquals("Paris is the capital of France.", passage.modelText());
  }

  @Test
  void preparedRagExposesPreparsedPassages() {
    PreparedRag prepared = RagFactory.of("Tokyo is the capital of Japan.");
    assertEquals(1, prepared.passages().size());
    assertFalse(prepared.passages().getFirst().termFreqs().isEmpty());
  }

  @Test
  void termCoveragePrefersFactCardOverTitleOnlyNarration() {
    PreparedRag prepared = RagFactory.builder()
        .options(RagLoadOptions.forTinyModels())
        .add("story.txt",
            "Little Red Riding Hood walked into the woods. The wolf thought of a tasty bite.")
        .add("facts.md", "Little Red Riding Hood is a fairy tale, not a fable.")
        .build();
    List<RagHit> hits = prepared.retrieve("Is Little Red Riding Hood a fairy tale or a fable?", 1);
    assertFalse(hits.isEmpty());
    assertTrue(hits.getFirst().chunk().text().toLowerCase().contains("fairy tale"));
  }

  @Test
  void bundledRagFranceQueryPrefersCapitals() {
    var root = com.igormaznitsa.nanollvm.utils.BundledRag.find();
    org.junit.jupiter.api.Assumptions.assumeTrue(root.isPresent(), "run tests from project root");
    PreparedRag prepared = RagFactory.make(root.get(), RagLoadOptions.forTinyModels());
    assertTrue(prepared.size() > 0);
    List<RagHit> hits = prepared.retrieve("capital of France Paris", 3);
    assertFalse(hits.isEmpty());
    assertTrue(hits.stream().anyMatch(hit ->
        hit.chunk().text().toLowerCase().contains("paris")));
  }
}
