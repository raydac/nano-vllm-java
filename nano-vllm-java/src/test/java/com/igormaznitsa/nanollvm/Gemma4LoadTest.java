package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class Gemma4LoadTest {

  @Test
  void loadsQatMobileAndGeneratesWhenWeightsPresent() {
    Path path = OptionalModelAssumptions.requireGemma4E2bQatMobile();
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model).maxModelLen(512).build()) {
      assertEquals(WeightNames.ARCH_GEMMA4, model.architectureName());
      assertTrue(model.modalities().accepts(LlmModality.TEXT));
      assertTrue(model.modalities().accepts(LlmModality.IMAGE));
      assertTrue(model.modalities().accepts(LlmModality.AUDIO));
      assertTrue(model.modalities().accepts(LlmModality.VIDEO));
      assertEquals(LlmModalities.TEXT_TO_TEXT, model.usableModalities());
      assertTrue(model.hfConfig().isGemma4());
      assertTrue(model.tokenizer().isTurnBasedChat());
      String prompt = model.tokenizer().applyChatTemplate(
        List.of(Map.of("role", "user", "content", "Say hi")), true, false);
      assertTrue(prompt.contains("<|turn>user"));
      assertTrue(prompt.contains("<|turn>model"));
      List<LLM.GenerationOutput> out = llm.generate(
        List.of(prompt),
        SamplingParams.builder().temperature(0.7f).maxTokens(8).build());
      assertEquals(1, out.size());
      assertFalse(out.getFirst().tokenIds().isEmpty());
    }
  }
}
