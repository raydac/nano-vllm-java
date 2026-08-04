package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.List;

/**
 * Immutable causal LM network graph (Qwen3, Gemma3, …). Weights are fixed at construction from a
 * {@link WeightBag}; there is no post-construction load or mutate API.
 */
public interface CausalLM {

  Tensor forward(Tensor inputIds, Tensor positions);

  Tensor computeLogits(Tensor hiddenStates);

  List<Attention> attentionLayers();

  String architectureName();
}
