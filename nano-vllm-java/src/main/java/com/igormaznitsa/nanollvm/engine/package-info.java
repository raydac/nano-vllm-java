/**
 * Continuous-batching engine for one {@link com.igormaznitsa.nanollvm.llm.LLM}: schedule a prefill
 * or decode batch, run the transformer, sample, then advance sequences.
 *
 * <p>Not exported. Application code drives this through {@code LLM.generate} / {@code chat} /
 * {@code complete}. One engine is not concurrent-safe for generate ({@code LLM.cancel()} is the
 * cross-thread abort).
 *
 * <p><b>Who owns what.</b> {@link Scheduler} admits work and pages KV via {@link BlockManager}.
 * {@link Transformer} owns the per-LLM {@link KvCacheArena} (and optional {@link ConvStateArena}),
 * binds them on an explicit {@link com.igormaznitsa.nanollvm.internal.Context} for one
 * {@link Transformer#step}, and samples. {@link Sequence} is one request's token stream, page
 * table, and sampling knobs.
 *
 * <p><b>Prefill vs decode.</b> Prefill packs newly scheduled prompt tokens and writes K/V.
 * Decode contributes the last token of each running sequence and attends over paged cache.
 * Prefill is preferred: {@link Scheduler#schedule()} returns a decode batch only when waiting
 * cannot proceed.
 */

package com.igormaznitsa.nanollvm.engine;
