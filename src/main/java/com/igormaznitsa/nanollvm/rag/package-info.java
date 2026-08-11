/**
 * Text RAG: load documents once into a shareable {@link PreparedRag} (chunking, BM25 index)
 * from paths, inline text, or classpath resources; optionally wrap with
 * {@link RagFactory#withEmbeddings} for hybrid dense retrieval, then {@link RagSession} for
 * retrieval-augmented chat.
 *
 * <p>{@link PreparedRag} is immutable and safe to share across threads; {@link RagSession} is not
 * thread-safe (one conversation thread). Dense indexes keep a live embedding model reference.
 */

package com.igormaznitsa.nanollvm.rag;
