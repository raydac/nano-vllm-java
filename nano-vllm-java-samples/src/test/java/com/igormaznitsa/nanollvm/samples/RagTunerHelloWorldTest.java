package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagResource;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import com.igormaznitsa.nanollvm.samples.utils.EpubText;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Locale;
import org.junit.jupiter.api.Test;

class RagTunerHelloWorldTest {

  private static String joinedHits(final PreparedRag corpus, final String query) {
    return corpus.retrieve(query, 5).stream()
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
  void epublibExtractsPlainTextFromBundledRur() {
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
  void ragTunerIndexesEpubAndRetrievesBookFacts() {
    PreparedRag corpus = RagTunerHelloWorld.indexBundledEpub();
    assertTrue(corpus.size() > 10);

    assertTrue(joinedHits(corpus, "Czech word robot meaning worker")
      .toLowerCase(Locale.ROOT)
      .contains("worker"));
    assertTrue(joinedHits(corpus, "Helena Glory Humanitarian League")
      .toLowerCase(Locale.ROOT)
      .contains("helena"));
  }
}
