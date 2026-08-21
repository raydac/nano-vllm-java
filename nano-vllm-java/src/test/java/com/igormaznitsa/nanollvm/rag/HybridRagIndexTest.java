package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class HybridRagIndexTest {

  private static TextChunk chunk(final String id, final String text) {
    return TextChunk.of(id, text);
  }

  private static RagHit hit(final String id, final double score) {
    return new RagHit(chunk(id, id), score);
  }

  @Test
  void ofFusesAnyIndexesByRrf() {
    FixedIndex first = new FixedIndex(2, false, List.of(hit("paris", 2.0), hit("wolf", 1.0)));
    FixedIndex second = new FixedIndex(2, false, List.of(hit("nile", 0.9), hit("paris", 0.8)));

    List<RagHit> fused = HybridRagIndex.of(first, second).retrieve("capital", 2);
    assertEquals(2, fused.size());
    assertEquals("paris", fused.getFirst().chunk().id());
  }

  @Test
  void ofFusesThreeIndexes() {
    TextChunk paris = chunk("paris", "Paris is the capital of France.");
    FixedIndex lexical = new FixedIndex(3, false,
      List.of(new RagHit(paris, 2.0), hit("wolf", 1.0)));
    FixedIndex dense = new FixedIndex(3, false,
      List.of(hit("nile", 0.9), new RagHit(paris, 0.8)));
    FixedIndex extra = new FixedIndex(3, false,
      List.of(new RagHit(paris, 0.7), hit("nile", 0.6)));

    List<RagHit> fused = HybridRagIndex.of(lexical, dense, extra).retrieve("capital", 1);
    assertEquals("paris", fused.getFirst().chunk().id());
    assertEquals(3, HybridRagIndex.of(lexical, dense, extra).indexes().size());
  }

  @Test
  void flattensNestedHybrids() {
    FixedIndex first = new FixedIndex(1, false, List.of(hit("a", 1.0)));
    FixedIndex second = new FixedIndex(1, false, List.of(hit("b", 1.0)));
    FixedIndex third = new FixedIndex(1, false, List.of(hit("c", 1.0)));
    HybridRagIndex nested = HybridRagIndex.of(first, second);

    HybridRagIndex fused = HybridRagIndex.of(nested, third);
    assertEquals(List.of(first, second, third), fused.indexes());
  }

  @Test
  void rejectsFewerThanTwoIndexes() {
    FixedIndex only = new FixedIndex(1, false, List.of(hit("a", 1.0)));
    assertThrows(IllegalArgumentException.class, () -> HybridRagIndex.of(List.of(only)));
    assertThrows(IllegalArgumentException.class, HybridRagIndex::of);
  }

  @Test
  void rejectsDuplicateIndexes() {
    FixedIndex index = new FixedIndex(1, false, List.of(hit("a", 1.0)));
    assertThrows(IllegalArgumentException.class, () -> HybridRagIndex.of(index, index));
  }

  @Test
  void isOutsideCorpusWhenEverySourceAgrees() {
    FixedIndex lexical = new FixedIndex(1, true, List.of());
    FixedIndex dense = new FixedIndex(1, true, List.of());
    FixedIndex extra = new FixedIndex(1, false, List.of());
    assertTrue(HybridRagIndex.of(lexical, dense).isOutsideCorpus("bmw"));
    assertFalse(HybridRagIndex.of(lexical, dense, extra).isOutsideCorpus("bmw"));
  }

  @Test
  void sizeWhenSourcesDisagreeIsUnknown() {
    FixedIndex small = new FixedIndex(2, false, List.of(hit("a", 1.0)));
    FixedIndex large = new FixedIndex(5, false, List.of(hit("b", 1.0)));
    assertEquals(-1, HybridRagIndex.of(small, large).size());
    assertEquals(2,
      HybridRagIndex.of(small, new FixedIndex(2, false, List.of(hit("c", 1.0)))).size());
  }

  @Test
  void lexicalAndDenseRequireThoseSources() {
    FixedIndex first = new FixedIndex(1, false, List.of(hit("a", 1.0)));
    FixedIndex second = new FixedIndex(1, false, List.of(hit("b", 1.0)));
    HybridRagIndex hybrid = HybridRagIndex.of(first, second);
    assertThrows(IllegalStateException.class, hybrid::lexical);
    assertThrows(IllegalStateException.class, hybrid::dense);
  }

  private record FixedIndex(int size, boolean outside, List<RagHit> hits) implements RagIndex {

    @Override
    public List<RagHit> retrieve(final String query, final int topK) {
      return this.hits.stream().limit(topK).toList();
    }

    @Override
    public boolean isOutsideCorpus(final String query) {
      return this.outside;
    }

    @Override
    public int size() {
      return this.size;
    }
  }
}
