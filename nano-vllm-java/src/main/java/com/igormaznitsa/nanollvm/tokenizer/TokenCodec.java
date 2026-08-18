package com.igormaznitsa.nanollvm.tokenizer;

import java.util.List;

/**
 * Encode/decode algorithm selected at tokenizer load.
 *
 * @since 1.1.1
 */
sealed interface TokenCodec
  permits Gpt2ByteBpeCodec, MetaspaceBpeCodec, WordPieceCodec, UnigramCodec, WordLevelCodec,
  CharCodec {

  List<Integer> encode(String text);

  String decode(List<String> pieces);
}
