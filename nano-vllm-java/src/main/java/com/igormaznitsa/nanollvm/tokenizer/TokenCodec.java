package com.igormaznitsa.nanollvm.tokenizer;

import java.util.List;

/**
 * Encode/decode algorithm selected at tokenizer load. Package-private behind
 * {@link Tokenizer}; application code never constructs a codec.
 *
 * @since 1.2.0
 */
sealed interface TokenCodec
  permits Gpt2ByteBpeCodec, MetaspaceBpeCodec, WordPieceCodec, UnigramCodec, WordLevelCodec,
  CharCodec {

  /**
   * Encodes {@code text} to vocabulary ids. Added/special tokens are matched as whole strings
   * before this codec's piece algorithm.
   *
   * @param text raw text; must not be {@code null}
   * @return token ids in order; never {@code null}
   */
  List<Integer> encode(String text);

  /**
   * Joins vocabulary pieces into a string (inverse of {@link #encode} for those pieces).
   *
   * @param pieces decoded token strings; must not be {@code null}
   * @return concatenated text; never {@code null}
   */
  String decode(List<String> pieces);
}
