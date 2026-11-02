package com.igormaznitsa.nanollvm.tensor.tornado;

import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoDeviceBudget.COMPACT_TILE_BYTES;
import static com.igormaznitsa.nanollvm.tensor.tornado.TornadoDeviceBudget.DEFAULT_HEAP_BYTES;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TornadoDeviceBudgetTest {

  @Test
  void defaultHeapKeepsTheCompactTile() {
    assertEquals(COMPACT_TILE_BYTES, TornadoDeviceBudget.weightBytes(DEFAULT_HEAP_BYTES, 0L, 0L));
    assertEquals(
      COMPACT_TILE_BYTES,
      TornadoDeviceBudget.weightBytes(DEFAULT_HEAP_BYTES, 16L * DEFAULT_HEAP_BYTES, 0L)
    );
  }

  @Test
  void missingHeapUsesTheDefaultTile() {
    assertEquals(COMPACT_TILE_BYTES, TornadoDeviceBudget.weightBytes(0L, 0L, 0L));
  }

  @Test
  void extraHeapAboveFourGibibytesHoldsOneMatrix() {
    long heap = DEFAULT_HEAP_BYTES + 2L * 1024 * 1024 * 1024;
    assertEquals(
      2L * 1024 * 1024 * 1024,
      TornadoDeviceBudget.weightBytes(heap, heap, 0L)
    );
  }

  @Test
  void deviceGlobalMemoryCapsAnOverstatedHeap() {
    assertEquals(
      COMPACT_TILE_BYTES,
      TornadoDeviceBudget.weightBytes(16L * DEFAULT_HEAP_BYTES, DEFAULT_HEAP_BYTES, 0L)
    );
  }

  @Test
  void configuredWeightAcceptsByteUnits() {
    assertEquals(32L * 1024 * 1024, TornadoDeviceBudget.parseWeight("32MB"));
    assertEquals(32L * 1024 * 1024, TornadoDeviceBudget.parseWeight("32MiB"));
    assertEquals(512L * 1024 * 1024, TornadoDeviceBudget.parseWeight("512M"));
    assertEquals(4096L, TornadoDeviceBudget.parseWeight("4096"));
  }

  @Test
  void configuredWeightRejectsOutOfRangeValues() {
    assertThrows(IllegalArgumentException.class, () -> TornadoDeviceBudget.parseWeight(" "));
    assertThrows(IllegalArgumentException.class, () -> TornadoDeviceBudget.parseWeight("32"));
    assertThrows(IllegalArgumentException.class, () -> TornadoDeviceBudget.parseWeight("128GB"));
  }

  @Test
  void layersPerGraphStaysInRange() {
    assertEquals(TornadoDecoder.DEFAULT_LAYERS, TornadoDecoder.layersPerGraph(null));
    assertEquals(1, TornadoDecoder.layersPerGraph("1"));
    assertEquals(TornadoDecoder.MAX_LAYERS, TornadoDecoder.layersPerGraph("16"));
    assertThrows(IllegalArgumentException.class, () -> TornadoDecoder.layersPerGraph("0"));
    assertThrows(IllegalArgumentException.class, () -> TornadoDecoder.layersPerGraph("17"));
    assertThrows(IllegalArgumentException.class, () -> TornadoDecoder.layersPerGraph("four"));
  }

  @Test
  void maximumAllocationCapsTheTile() {
    long maxAllocation = 8L * 1024 * 1024;
    assertEquals(
      maxAllocation,
      TornadoDeviceBudget.weightBytes(DEFAULT_HEAP_BYTES, DEFAULT_HEAP_BYTES, maxAllocation)
    );
    assertEquals(
      maxAllocation,
      TornadoDeviceBudget.weightBytes(2L * DEFAULT_HEAP_BYTES, 2L * DEFAULT_HEAP_BYTES,
        maxAllocation)
    );
  }
}
