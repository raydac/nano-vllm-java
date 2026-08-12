package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.List;

/**
 * Immutable causal LM network graph (Qwen3, Gemma3, …). Weights are fixed at construction from a
 * {@link WeightBag}; there is no post-construction load or mutate API.
 */
public interface CausalLM {

  Tensor forward(final Tensor inputIds, final Tensor positions, final Context context);

  Tensor computeLogits(final Tensor hiddenStates, final Context context);

  List<Attention> attentionLayers();

  String architectureName();
}
