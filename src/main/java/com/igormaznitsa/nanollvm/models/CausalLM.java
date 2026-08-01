package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.List;
import java.util.Map;

/**
 * Pluggable causal LM backend (Qwen3, Gemma3, …).
 */
public interface CausalLM {

  Tensor forward(Tensor inputIds, Tensor positions);

  Tensor computeLogits(Tensor hiddenStates);

  List<Attention> attentionLayers();

  WeightSlot getParameter(String name);

  boolean hasParameter(String name);

  Map<String, Object[]> packedModulesMapping();

  String architectureName();

  @FunctionalInterface
  interface WeightSlot {
    static WeightSlot of(java.util.function.Consumer<Tensor> loader) {
      return (tensor, shardId) -> loader.accept(tensor);
    }

    static WeightSlot qkv(com.igormaznitsa.nanollvm.layers.Linear.Qkv linear) {
      return (tensor, shardId) -> linear.loadShard(tensor, String.valueOf(shardId));
    }

    static WeightSlot merged(com.igormaznitsa.nanollvm.layers.Linear.Merged linear) {
      return (tensor, shardId) -> linear.loadShard(tensor, ((Number) shardId).intValue());
    }

    void load(Tensor tensor, Object shardId);
  }
}
