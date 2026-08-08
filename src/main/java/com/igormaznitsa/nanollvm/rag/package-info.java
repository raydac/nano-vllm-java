/**
 * Text RAG: load documents once into a shareable {@link PreparedRag} (chunking, BM25 index),
 * then {@link RagSession} for retrieval-augmented chat.
 *
 * <p>{@link PreparedRag} is immutable and safe to share across threads; {@link RagSession} is not
 * thread-safe (one conversation thread).
 */
package com.igormaznitsa.nanollvm.rag;
