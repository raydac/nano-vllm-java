package com.igormaznitsa.nanollvm;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.createTempFile;
import static java.nio.file.Files.deleteIfExists;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.rag.Bm25Index;
import com.igormaznitsa.nanollvm.rag.PassagePreparser;
import com.igormaznitsa.nanollvm.rag.PreparedPassage;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagPrompt;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import com.igormaznitsa.nanollvm.rag.TextCorpus;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class RagUnitTest {

  @Test
  void bm25PrefersMatchingString() {
    Bm25Index index = Bm25Index.of(
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
      Bm25Index fromFile = Bm25Index.fromFile(a);
      assertEquals(1, fromFile.size());
      assertTrue(fromFile.retrieve("Nile river", 1).getFirst().chunk().text().contains("Nile"));

      Bm25Index fromFolder = Bm25Index.fromFolder(dir);
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
      TextCorpus corpus = TextCorpus.builder()
          .add("id-earth", "Earth is the third planet from the Sun.")
          .addFile(file)
          .addFolder(dir)
          .build();
      Bm25Index index = Bm25Index.build(corpus);
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
    Bm25Index index = Bm25Index.of("Cats are mammals.");
    List<RagHit> hits = index.retrieve("mammals", 1);
    String prompt = RagPrompt.format(hits, "What are cats?");
    assertTrue(prompt.contains("Cats are mammals."));
    assertTrue(prompt.contains("What are cats?"));
    assertTrue(prompt.contains("Context:"));
  }

  @Test
  void compactPromptPutsQuestionBeforePassages() {
    Bm25Index index = Bm25Index.of("The Nile is a major river in Africa.");
    List<RagHit> hits = index.retrieve("Nile", 1);
    String prompt = RagPrompt.format(hits, "what is Nile?", 900, true);
    assertTrue(prompt.contains("The Nile is a major river in Africa."));
    assertTrue(prompt.contains("what is Nile?"));
    assertTrue(prompt.indexOf("what is Nile?") < prompt.indexOf("The Nile is a major river"));
    assertTrue(prompt.contains("Notes (use only if relevant"));
    assertFalse(prompt.contains("If the context is insufficient"));
    assertFalse(prompt.contains("previous answer was wrong"));
  }

  @Test
  void relativeScoreFilterDropsWeakSecondHit() {
    Bm25Index index = Bm25Index.of(
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
    assertFalse(prepared.corpus().chunks().isEmpty());
    String joined = prepared.corpus().chunks().stream()
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
    TextChunk chunk = new TextChunk(
        "id",
        "/docs/facts-capitals.md",
        "Paris is the capital of France.");
    PreparedPassage passage = PassagePreparser.prepareOne(chunk);
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
