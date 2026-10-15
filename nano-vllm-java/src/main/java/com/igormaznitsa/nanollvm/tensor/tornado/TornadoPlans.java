package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

/**
 * Precompiled TornadoVM plans that retain only the latest execution result.
 *
 * @since 1.5.0
 */
final class TornadoPlans {

  private TornadoPlans() {
  }

  static TornadoExecutionPlan open(final ImmutableTaskGraph snapshot) {
    return new TornadoExecutionPlan(snapshot)
      .withPreCompilation()
      .withoutPlanResultsHistory();
  }
}
