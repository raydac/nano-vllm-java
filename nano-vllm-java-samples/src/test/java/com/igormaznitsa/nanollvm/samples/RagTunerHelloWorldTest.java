package com.igormaznitsa.nanollvm.samples;

import static java.util.stream.Collectors.joining;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.DenseRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagResource;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import com.igormaznitsa.nanollvm.samples.utils.EpubText;
import com.igormaznitsa.nanollvm.samples.utils.SampleModelAssumptions;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Test;

class RagTunerHelloWorldTest {

  private static String joinedHits(final DenseRagIndex index, final String query) {
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

  private static String bookExcerpt() {
    String text = EpubText.extract(loadedEpub()).orElseThrow();
    return text.substring(0, Math.min(text.length(), 8_000));
  }

  @Test
  void ragTunerChunksBundledEpub() {
    PreparedRag documents = RagTunerHelloWorld.loadBundledEpub();
    assertTrue(documents.size() > 10);
  }

  @Test
  void denseIndexRetrievesBookFactsWhenGtePresent() {
    Path gte = SampleModelAssumptions.requireGteSmallGguf();
    String excerpt = bookExcerpt();
    PreparedRag documents = RagFactory.builder()
      .add("rur-excerpt", excerpt)
      .build();

    try (LlmModel embed = LlmModelFactory.make(gte)) {
      DenseRagIndex index = DenseRagIndex.of(documents, embed);
      assertTrue(index.size() > 0);

      assertTrue(joinedHits(index, "What does the Czech word robot mean?")
        .toLowerCase(Locale.ROOT)
        .contains("worker"));
      assertTrue(joinedHits(index, "Who is Helena Glory?")
        .toLowerCase(Locale.ROOT)
        .contains("helena"));
    }
  }
}
