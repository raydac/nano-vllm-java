package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Cuts one loaded document into {@link TextChunk}s. Register it with
 * {@link RagFactory.Builder#splitter(RagSplitter)} before adding documents. When none is set, the
 * loader uses {@link #standard(RagLoadOptions)} for the builder's {@link RagLoadOptions}.
 *
 * <p>The loader calls {@link #split(RagDocument)} once per document, after
 * {@link RagTuner} extract and preprocess. Returned chunks are indexed as they are: chunk-size
 * and sentence knobs on {@link RagLoadOptions} are not applied again. Blank chunks are dropped.
 * Chunk {@linkplain TextChunk#id() ids} must be unique in the corpus. Dedupe still follows
 * {@link RagLoadOptions#dedupe()}.
 *
 * <pre>{@code
 * PreparedRag rag = RagFactory.builder()
 *     .splitter(document -> paragraphs(document))
 *     .addFolder(Path.of("docs"))
 *     .build();
 * }</pre>
 *
 * @since 1.5.0
 */
@FunctionalInterface
public interface RagSplitter {

  /**
   * Built-in cutter: sentence packing or a sliding window, from {@code options}.
   *
   * @param options chunk ceiling, overlap, and sentence knobs; must not be {@code null}
   * @return a splitter that delegates to the loader's default cut
   * @throws NullPointerException if {@code options} is {@code null}
   */
  static RagSplitter standard(final RagLoadOptions options) {
    requireNonNull(options, "options");
    return document -> CorpusLoader.split(requireNonNull(document, "document"), options);
  }

  /**
   * Cuts {@code document} into passages. An empty list indexes nothing from that document.
   *
   * @param document body after preprocess; must not be {@code null}
   * @return chunks for this document; must not be {@code null} and must not contain {@code null}
   * @throws NullPointerException if {@code document} is {@code null}
   */
  List<TextChunk> split(RagDocument document);
}
