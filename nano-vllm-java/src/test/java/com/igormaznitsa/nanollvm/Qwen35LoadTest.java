package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class Qwen35LoadTest {

  @Test
  void loadsJeffTextAndGeneratesWhenWeightsPresent() {
    Path path = OptionalModelAssumptions.requireJeffQwen35();
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model).maxModelLen(128).build()) {
      assertEquals(WeightNames.ARCH_QWEN3_5, model.architectureName());
      assertTrue(model.isCausalModel());
      assertTrue(model.hfConfig().isQwen35());
      List<LLM.GenerationOutput> out = llm.generate(
        List.of("Hi"),
        SamplingParams.builder().temperature(0.7f).maxTokens(1).build());
      assertEquals(1, out.size());
      assertFalse(out.getFirst().tokenIds().isEmpty());
    }
  }
}
