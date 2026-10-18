package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import java.util.Map;

final class WeightLayers {

  private WeightLayers() {
  }

  static Linear.Row linearRow(final WeightBag weights, final String name) {
    return weights.findPacked(name)
      .map(Linear.Row::new)
      .orElseGet(() -> new Linear.Row(weights.require(name)));
  }

  static VocabParallelEmbedding embedding(final WeightBag weights, final String name) {
    return weights.findPacked(name)
      .map(VocabParallelEmbedding::new)
      .orElseGet(() -> new VocabParallelEmbedding(weights.require(name)));
  }

  static ParallelLMHead tiedLmHead(final VocabParallelEmbedding embed) {
    return embed.isPacked()
      ? new ParallelLMHead(embed.packedWeight())
      : new ParallelLMHead(embed.weight());
  }

  static float ropeTheta(final Config.HfConfig config) {
    float theta = config.ropeTheta();
    Map<String, Object> scaling = config.ropeScaling();
    if (scaling.containsKey("rope_theta")) {
      return Json.asFloat(scaling.get("rope_theta"), theta);
    }
    return theta;
  }
}
