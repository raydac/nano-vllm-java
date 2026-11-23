package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

/**
 * One phonemized utterance: inventory ids and a parallel stress feature (0, 1, or 2).
 *
 * @param ids    token ids, including the edge marks
 * @param stress stress aligned with {@code ids}
 */
public record TinyVocaUtterance(int[] ids, int[] stress) {

  public TinyVocaUtterance {
    ids = ids.clone();
    stress = stress.clone();
    if (ids.length != stress.length || ids.length == 0) {
      throw new IllegalArgumentException("ids and stress must be the same non-empty length");
    }
  }

  @Override
  public int[] ids() {
    return this.ids.clone();
  }

  @Override
  public int[] stress() {
    return this.stress.clone();
  }
}
