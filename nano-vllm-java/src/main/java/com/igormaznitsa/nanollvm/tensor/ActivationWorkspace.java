package com.igormaznitsa.nanollvm.tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reuses large activation {@code float[]} buffers for one transformer step.
 *
 * <p>Bound on the generate thread for {@code Transformer} step / prefill. Each layer keeps the
 * tensors it still needs and returns the rest to a freelist keyed by length. Weights and any
 * tensor allocated while unbound stay on the ordinary heap.
 */
public final class ActivationWorkspace {

  static final int MIN_LENGTH = 256;

  private static final int MAX_PARKED_FLOATS = 128 * 1024 * 1024;

  private static final ThreadLocal<ActivationWorkspace> CURRENT = new ThreadLocal<>();

  private final Map<Integer, ArrayDeque<float[]>> free = new HashMap<>();
  private final List<Lease> live = new ArrayList<>();
  private final List<Lease> layer = new ArrayList<>();
  private boolean inLayer;
  private int parkedFloats;

  public static void bind(final ActivationWorkspace workspace) {
    CURRENT.set(workspace);
  }

  public static void unbind() {
    CURRENT.remove();
  }

  static ActivationWorkspace current() {
    return CURRENT.get();
  }

  /**
   * Starts a layer. Allocations until {@link #endLayer} belong to that layer.
   */
  public static void beginLayer() {
    ActivationWorkspace workspace = CURRENT.get();
    if (workspace != null) {
      workspace.inLayer = true;
      workspace.layer.clear();
    }
  }

  /**
   * Returns this layer's buffers except {@code keep}. Shared storage (a reshape of a kept
   * tensor) stays live.
   */
  public static void endLayer(final Tensor... keep) {
    ActivationWorkspace workspace = CURRENT.get();
    if (workspace == null) {
      return;
    }
    if (keep != null) {
      for (Tensor tensor : keep) {
        workspace.markKept(tensor);
      }
    }
    for (Lease lease : workspace.layer) {
      if (!lease.kept && !lease.detached) {
        workspace.recycle(lease);
      }
    }
    workspace.layer.clear();
    workspace.inLayer = false;
  }

  /**
   * Returns {@code old} when none of {@code live} still uses its storage.
   */
  public static void releaseUnless(final Tensor old, final Tensor... live) {
    ActivationWorkspace workspace = CURRENT.get();
    if (workspace == null || old == null) {
      return;
    }
    Lease lease = old.lease();
    if (lease == null || lease.recycled || lease.detached) {
      return;
    }
    if (live != null) {
      for (Tensor tensor : live) {
        if (tensor != null && tensor.lease() == lease) {
          return;
        }
      }
    }
    workspace.recycle(lease);
  }

  /**
   * Leaves {@code tensor}'s storage with the caller. {@link #recycleAll()} will not reuse it.
   */
  public static void detach(final Tensor tensor) {
    if (tensor == null) {
      return;
    }
    Lease lease = tensor.lease();
    if (lease != null) {
      lease.detached = true;
    }
  }

  public void recycleAll() {
    for (Lease lease : this.live) {
      if (!lease.recycled && !lease.detached) {
        this.park(lease.data);
        lease.recycled = true;
      }
    }
    this.live.clear();
    this.layer.clear();
    this.inLayer = false;
  }

  Tensor allocate(final int[] shape, final int length) {
    float[] data = this.checkout(length);
    Lease lease = new Lease(data);
    this.live.add(lease);
    if (this.inLayer) {
      this.layer.add(lease);
    }
    return Tensor.pooled(data, shape, lease);
  }

  private void markKept(final Tensor tensor) {
    if (tensor == null) {
      return;
    }
    Lease lease = tensor.lease();
    if (lease != null) {
      lease.kept = true;
    }
  }

  private void recycle(final Lease lease) {
    if (lease.recycled || lease.detached) {
      return;
    }
    lease.recycled = true;
    this.park(lease.data);
  }

  private float[] checkout(final int length) {
    ArrayDeque<float[]> bin = this.free.get(length);
    if (bin == null || bin.isEmpty()) {
      return new float[length];
    }
    this.parkedFloats -= length;
    float[] data = bin.removeFirst();
    Arrays.fill(data, 0f);
    return data;
  }

  private void park(final float[] data) {
    int length = data.length;
    if (this.parkedFloats + length > MAX_PARKED_FLOATS) {
      return;
    }
    this.free.computeIfAbsent(length, ignored -> new ArrayDeque<>()).addFirst(data);
    this.parkedFloats += length;
  }

  static final class Lease {
    final float[] data;
    boolean kept;
    boolean recycled;
    boolean detached;

    Lease(final float[] data) {
      this.data = data;
    }
  }
}
