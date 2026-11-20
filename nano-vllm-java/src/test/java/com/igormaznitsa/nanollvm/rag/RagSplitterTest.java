package com.igormaznitsa.nanollvm.rag;

import static java.nio.file.Files.createTempFile;
import static java.nio.file.Files.deleteIfExists;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * Custom {@link RagSplitter} on {@link RagFactory.Builder}.
 */
class RagSplitterTest {

  private static final Pattern HIDDEN_BLOCK = Pattern.compile(
    "<(script|style)\\b[^>]*>.*?</\\1>", Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
  private static final Pattern TEXT_BLOCK = Pattern.compile(
    "<(title|h[1-6]|p|li)\\b[^>]*>(.*?)</\\1>", Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
  private static final Pattern TAG = Pattern.compile("<[^>]+>");
  private static final Pattern SPACE = Pattern.compile("\\s+");
  private static final Pattern SPACE_BEFORE_PUNCT = Pattern.compile("\\s+([.,;:!?])");

  private static final String CITY_GUIDE = """
    <html>
      <head>
        <title>Cities</title>
        <style>p { color: red; }</style>
      </head>
      <body>
        <h1>City guide</h1>
        <p>Paris is the capital of France.</p>
        <p>Berlin is a city in <b>Germany</b>.</p>
        <ul><li>Lyon is in France.</li></ul>
        <script>var secret = "do not index";</script>
      </body>
    </html>
    """;

  private static List<TextChunk> htmlBlocks(final RagDocument document) {
    String visible = HIDDEN_BLOCK.matcher(document.text()).replaceAll("");
    Matcher blocks = TEXT_BLOCK.matcher(visible);
    List<TextChunk> chunks = new ArrayList<>();
    while (blocks.find()) {
      String text = plainText(blocks.group(2));
      if (text.isEmpty()) {
        continue;
      }
      chunks.add(new TextChunk(
        document.id() + "#" + (chunks.size() + 1), document.source(), text));
    }
    return List.copyOf(chunks);
  }

  private static String plainText(final String markup) {
    String text = SPACE.matcher(TAG.matcher(markup).replaceAll(" ")).replaceAll(" ");
    return SPACE_BEFORE_PUNCT.matcher(text).replaceAll("$1").strip();
  }

  @Test
  void htmlDocumentIsSplitIntoPlainTextBlocks() {
    PreparedRag rag = RagFactory.builder()
      .splitter(RagSplitterTest::htmlBlocks)
      .add("guide", CITY_GUIDE)
      .build();

    assertEquals(List.of(
      "Cities",
      "City guide",
      "Paris is the capital of France.",
      "Berlin is a city in Germany.",
      "Lyon is in France."
    ), rag.chunks().stream().map(TextChunk::text).toList());
    assertEquals(List.of("guide#1", "guide#2", "guide#3", "guide#4", "guide#5"),
      rag.chunks().stream().map(TextChunk::id).toList());
    assertTrue(rag.chunks().stream().allMatch(chunk -> chunk.source().equals("guide")));
    assertTrue(rag.chunks().stream().noneMatch(chunk -> chunk.text().contains("<")
      || chunk.text().contains("do not index")
      || chunk.text().contains("color")));
    assertTrue(rag.retrieve("Paris France capital", 1).getFirst().chunk().text()
      .contains("Paris"));

    PreparedRag builtIn = RagFactory.builder().add("guide", CITY_GUIDE).build();
    assertTrue(builtIn.chunks().stream().anyMatch(chunk -> chunk.text().contains("<")));
  }

  @Test
  void splitterReplacesTheBuiltInCut() {
    PreparedRag rag = RagFactory.builder()
      .splitter(document -> List.of(
        new TextChunk(document.id() + "#a", document.source(), "alpha fact"),
        new TextChunk(document.id() + "#b", document.source(), "beta fact")))
      .add("doc", "this body is not the built-in window")
      .build();

    assertEquals(List.of("alpha fact", "beta fact"),
      rag.chunks().stream().map(TextChunk::text).toList());
    assertEquals("doc", rag.chunks().getFirst().source());
  }

  @Test
  void splitterSeesPreprocessedText() {
    PreparedRag rag = RagFactory.builder()
      .addProcessor(RagTuner.preprocessing(text -> "ONLY"))
      .splitter(document -> List.of(TextChunk.of(document.id(), document.text())))
      .add("doc", "original")
      .build();

    assertEquals("ONLY", rag.chunks().getFirst().text());
  }

  @Test
  void emptySplitDropsThatDocument() {
    PreparedRag rag = RagFactory.builder()
      .splitter(document -> document.text().contains("keep")
        ? List.of(TextChunk.of(document.id(), document.text()))
        : List.of())
      .add("keep", "keep this passage")
      .add("drop", "drop this passage")
      .build();

    assertEquals(1, rag.size());
    assertTrue(rag.chunks().getFirst().text().contains("keep"));
  }

  @Test
  void blankChunksAreDropped() {
    PreparedRag rag = RagFactory.builder()
      .splitter(document -> List.of(
        TextChunk.of(document.id() + "#blank", "  "),
        TextChunk.of(document.id() + "#kept", "kept passage")))
      .add("doc", "body")
      .build();

    assertEquals(1, rag.size());
    assertEquals("kept passage", rag.chunks().getFirst().text());
  }

  @Test
  void duplicateChunkIdIsRejected() {
    IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
      () -> RagFactory.builder()
        .splitter(document -> List.of(
          TextChunk.of("same", "one passage"),
          TextChunk.of("same", "two passage")))
        .add("body")
        .build());

    assertTrue(failure.getMessage().contains("same"));
  }

  @Test
  void duplicateIdAcrossDocumentsIsRejected() {
    RagSplitter fixed = document -> List.of(TextChunk.of("fixed", document.text()));

    assertThrows(IllegalArgumentException.class, () -> RagFactory.builder()
      .splitter(fixed)
      .add("one passage")
      .add("two passage")
      .build());
  }

  @Test
  void nullSplitIsRejected() {
    assertThrows(NullPointerException.class, () -> RagFactory.builder()
      .splitter(document -> null)
      .add("body")
      .build());
  }

  @Test
  void splitterMustBeSetBeforeDocuments() {
    RagFactory.Builder builder = RagFactory.builder().add("hello passage");

    assertThrows(IllegalStateException.class,
      () -> builder.splitter(document -> List.of(TextChunk.of(document.id(), document.text()))));
  }

  @Test
  void splitterReceivesTheFileSource() throws Exception {
    Path file = createTempFile("nanollvm-rag-split", ".txt");
    writeString(file, "hello from the file");
    AtomicReference<String> source = new AtomicReference<>();
    try {
      PreparedRag rag = RagFactory.builder()
        .splitter(document -> {
          source.set(document.source());
          return List.of(TextChunk.of("file-chunk", document.text()));
        })
        .addFile(file)
        .build();

      assertTrue(source.get().endsWith(file.getFileName().toString()));
      assertTrue(rag.chunks().getFirst().text().contains("hello from the file"));
    } finally {
      deleteIfExists(file);
    }
  }

  @Test
  void standardSplitterHonorsTheCeiling() {
    RagLoadOptions options = RagLoadOptions.defaults()
      .withPreprocess(false)
      .withMaxChunkChars(64)
      .withChunkOverlap(0);
    PreparedRag rag = RagFactory.builder()
      .options(options)
      .splitter(RagSplitter.standard(options))
      .add("doc", "a".repeat(130))
      .build();

    assertTrue(rag.chunks().size() >= 2);
    assertTrue(rag.chunks().stream().allMatch(chunk -> chunk.text().length() <= 64));
  }
}
