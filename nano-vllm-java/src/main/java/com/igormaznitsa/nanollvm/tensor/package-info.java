/**
 * Non-exported numeric core: {@link Tensor} storage, {@link Ops} layer bricks, and
 * {@link MatmulRuntime} for one engine's CPU workers.
 *
 * <p>{@link LinearKernel} / {@link EmbeddingKernel} bind one weight table at construction
 * (dense float32 or packed GGUF). {@link FloatKernels} is the pluggable scalar/SIMD backend.
 * Subpackages {@code tensor.kernels}, {@code tensor.scalar}, and {@code tensor.vector} are the
 * implementations. Application code stays on {@link com.igormaznitsa.nanollvm.llm.LLM.Builder}
 * ({@code cpuThreads}, {@code dedicatedMatmulPool}).
 */

package com.igormaznitsa.nanollvm.tensor;
