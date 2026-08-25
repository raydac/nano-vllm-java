/**
 * Non-exported runtime helpers: per-step {@link Context}, in-memory {@link ModelFileBundle},
 * GGUF dequant tables, and a small JSON reader for {@code config.json} / tokenizer sidecars.
 *
 * <p>Application code does not import this package. {@link Context} is owned by
 * {@link com.igormaznitsa.nanollvm.engine.Transformer} for one forward. {@link ModelFileBundle}
 * is the heap snapshot of a {@link com.igormaznitsa.nanollvm.models.ModelFileSource} load.
 */

package com.igormaznitsa.nanollvm.internal;
