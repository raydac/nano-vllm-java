package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagResource;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import com.igormaznitsa.nanollvm.samples.utils.EpubText;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Test;

class RagTunerHelloWorldTest {

  private static String joinedHits(final RagIndex index, final String query) {
    return index.retrieve(query, 5).stream()
      .map(RagHit::chunk)
      .map(TextChunk::text)
      .collect(joining(" "));
  }

  private static RagResource loadedEpub() {
    try (InputStream in = RagTunerHelloWorld.class.getResourceAsStream("/" + EpubText.RUR_EPUB)) {
      if (in == null) {
        throw new IllegalStateException("missing classpath resource /" + EpubText.RUR_EPUB);
      }
      return RagResource.classpath(EpubText.RUR_EPUB, in.readAllBytes());
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @Test
  void extractsPlainTextFromBundledRur() {
    String text = EpubText.extract(loadedEpub()).orElseThrow();
    String folded = text.toLowerCase(Locale.ROOT);

    assertTrue(folded.contains("rossum's universal robots")
      || folded.contains("rossum’s universal robots"));
    assertTrue(folded.contains("karel"));
    assertTrue(folded.contains("helena glory"));
    assertTrue(folded.contains("worker"));
    assertFalse(folded.contains("<html"));
  }

  @Test
  void ragTunerChunksBundledEpub() {
    PreparedRag documents = RagTunerHelloWorld.loadBundledEpub();
    assertTrue(documents.size() > 10);
  }

  @Test
  void helenaGloryChunksCarryPlayTitle() {
    List<String> helena = RagTunerHelloWorld.loadBundledEpub().chunks().stream()
      .map(TextChunk::text)
      .map(text -> text.toLowerCase(Locale.ROOT))
      .filter(text -> text.contains("helena glory"))
      .toList();
    assertFalse(helena.isEmpty());
    assertTrue(helena.stream().anyMatch(text -> text.contains("r.u.r")
      || text.contains("rossum")
      || text.contains("čapek")
      || text.contains("capek")));
    assertTrue(helena.stream().noneMatch(text -> text.contains("fantastic melodrama")));
  }

  @Test
  void bm25RetrievesBookFacts() {
    PreparedRag documents = RagTunerHelloWorld.loadBundledEpub();

    assertTrue(joinedHits(documents, "What does the Czech word robot mean?")
      .toLowerCase(Locale.ROOT)
      .contains("worker"));
    assertTrue(joinedHits(documents, "Who is Helena Glory?")
      .toLowerCase(Locale.ROOT)
      .contains("helena"));
  }
}
