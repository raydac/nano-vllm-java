/**
 * Text-based retrieval-augmented generation.
 *
 * <p>{@link RagFactory} loads documents once into a shareable {@link PreparedRag}:
 * section-aware sentences, {@link PassagePreparser} (model vs search text, pre-tokenized TF),
 * and an inverted {@link Bm25Index}. Sessions only query that prepared index.
 * Plain-text and markup files are read as UTF-8; {@code .pdf} files go through
 * {@link PdfTextExtractor} (pure Java; no OCR).
 */

package com.igormaznitsa.nanollvm.rag;
