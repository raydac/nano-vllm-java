package com.igormaznitsa.nanollvm.tokenizer;

import java.util.List;

sealed interface TokenCodec
  permits Gpt2ByteBpeCodec, MetaspaceBpeCodec, WordPieceCodec, UnigramCodec, WordLevelCodec,
  CharCodec {

  List<Integer> encode(String text);

  String decode(List<String> pieces);
}
