/**
 * Transformer layer bricks used by architecture graphs in {@code models.internal}. Not exported.
 *
 * <p>Each type owns <em>weights plus a forward</em> for one operation: causal or encoder attention,
 * affine maps, token embeddings, RMS/LayerNorm, rotary embeddings, LFM2 short-conv, and next-token
 * sampling. Step state (paged KV, conv arena, matmul runtime, varlen metadata) arrives on an
 * explicit {@link com.igormaznitsa.nanollvm.internal.Context} owned by
 * {@link com.igormaznitsa.nanollvm.engine.Transformer}. Application code does not construct these;
 * {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} builds the graph.
 *
 * <p><b>Causal vs encoder.</b> {@link Attention} is grouped-query attention with a paged KV cache
 * (chat). {@link BidirectionalAttention} is full self-attention with no cache (BERT embeddings).
 * Independent attention jobs, RoPE tokens, and embedding rows may run on the step
 * {@link com.igormaznitsa.nanollvm.internal.Context#matmul()} pool. Decoder layers stay sequential
 * (residual stream).
 *
 * <p><b>Weights.</b> {@link Linear} and {@link VocabParallelEmbedding} bind a dense float32 table,
 * a packed GGUF weight, or Gemma QAT at construction. {@code VocabParallelEmbedding} keeps the vLLM
 * name; this port is single-device (no tensor parallel).
 */

package com.igormaznitsa.nanollvm.layers;
