package com.igormaznitsa.nanollvm.chat;

import com.igormaznitsa.nanollvm.llm.LLM;

@FunctionalInterface
public interface LlmListener {

  void onText(LLM source, LlmTextEvent event);
}
