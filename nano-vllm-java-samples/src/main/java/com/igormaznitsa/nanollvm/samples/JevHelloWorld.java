package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.jev.Jev;
import com.igormaznitsa.nanollvm.jev.JevDecision;
import com.igormaznitsa.nanollvm.jev.JevLevel;
import com.igormaznitsa.nanollvm.jev.JevOption;
import com.igormaznitsa.nanollvm.jev.JevQuestion;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Tour of typed decisions on one causal chat model.
 *
 * <p>A choice is read at L0 (every rotation of the options) and again at RAW (one prefill, options
 * in the order given). A yes/no and an ordered score follow. The last section decides the same
 * yes/no eight times so the label prior turns on, then repeats two of those states so the change
 * in probability is on the page. Each rotation is one prefill, so a four-way L0 choice costs four
 * prefills.
 *
 * <p>Args: optional chat checkpoint (default the smallest local chat demo via
 * {@link BundledModels#requireChatDemo()}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.JevHelloWorld}
 *
 * @since 1.5.0
 */
public final class JevHelloWorld {

  private static final JevQuestion ROUTE = JevQuestion.choice(
    "Which team should handle this?",
    List.of("billing", "technical", "sales", "other")).named("route");

  private static final String ROUTE_STATE = "My card was charged twice for one order.";

  private static final JevQuestion FAILURE = JevQuestion.yesNo("Is this a technical failure?");

  private static final String FAILURE_STATE =
    "The payment page returns 500 and the customer cannot check out.";

  private static final JevQuestion URGENCY = JevQuestion.score(
    "How urgent is the customer's request?",
    List.of("low", "medium", "high")).named("urgency");

  private static final JevQuestion REFUND =
    JevQuestion.yesNo("Does this message ask for a refund?");

  private static final String MONEY_BACK = "I want my money back for order 441.";

  private static final String PACKAGE_DUE = "Where is my package? It was due Monday.";

  private static final List<String> REFUND_STATES = List.of(
    MONEY_BACK,
    "Please refund the duplicate charge.",
    PACKAGE_DUE,
    "Can you change the delivery address?",
    "The item arrived broken. I need a refund.",
    "I only wanted to track the shipment.",
    "Cancel the order and return the payment.",
    "What are your store hours?"
  );

  private static final List<String> PRIOR_REPEATS = List.of(PACKAGE_DUE, MONEY_BACK);

  private JevHelloWorld() {
  }

  public static void main(final String[] args) {
    Path path = checkpoint(args);
    explain(path);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(path);
         Jev l0 = LLM.builder(model).toJev();
         Jev raw = LLM.builder(model).jevLevel(JevLevel.RAW).toJev()) {
      System.out.println("architecture=" + model.architectureName());
      System.out.println("Two engines share this model. Closing a Jev closes its engine only.");
      showChoice(l0, raw);
      showYesNo(l0);
      showScore(l0);
      showPrior(l0);
    } finally {
      System.out.println();
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static void explain(final Path path) {
    System.out.println("Typed decisions from " + path);
    System.out.println("The model does not write a sentence. The prompt ends where the answer");
    System.out.println("label would be, and the decision is the probability of those labels.");
    System.out.println("  choice  letters A, B, C, …     L0 reads one prefill per option");
    System.out.println("  yes/no  the words Yes and No   L0 reads both phrasings");
    System.out.println(
      "  score   ordered levels         one prefill; score() is the expected level");
    System.out.println("  RAW     one prefill, options in the order given");
    System.out.println("  L0      rotations averaged in log space; label prior after 8 states");
    System.out.println("Columns: rotations read, whether the prior corrected this answer,");
    System.out.println("how often a later rotation disagreed with the first (orderFlip),");
    System.out.println("and how much probability sat on the labels at all (labelMass).");
  }

  private static void showChoice(final Jev l0, final Jev raw) {
    heading("Choice — L0 rotates the four options, RAW keeps the order above");
    System.out.println("state: " + ROUTE_STATE);
    System.out.println("question: " + ROUTE.text());
    System.out.println("options: " + ROUTE.options());
    System.out.println("reading L0 (4 prefills)…");
    JevDecision rotated = l0.decide(ROUTE_STATE, ROUTE);
    printDecision(rotated);
    showRequire(rotated);

    System.out.println("reading RAW (1 prefill, same option order)…");
    JevDecision once = raw.decide(ROUTE_STATE, ROUTE);
    printDecision(once);
    showRequire(once);
  }

  private static void showYesNo(final Jev l0) {
    heading("Yes/no — L0 asks \"Yes or No\" and \"No or Yes\"");
    System.out.println("state: " + FAILURE_STATE);
    System.out.println("question: " + FAILURE.text());
    System.out.println("reading L0 (2 prefills)…");
    printDecision(l0.decide(FAILURE_STATE, FAILURE));
  }

  private static void showScore(final Jev l0) {
    heading("Score — ordered levels stay in one order, so L0 is still one prefill");
    System.out.println("state: " + FAILURE_STATE);
    System.out.println("question: " + URGENCY.text());
    System.out.println("levels: " + URGENCY.options() + "  values " + URGENCY.binCenters());
    System.out.println("reading…");
    JevDecision decision = l0.decide(FAILURE_STATE, URGENCY);
    printDecision(decision);
    decision.score().ifPresent(value ->
      System.out.println("  expected level " + format(value) + "  (0=low, 1=medium, 2=high)"));
  }

  private static void showPrior(final Jev jev) {
    heading("Label prior — same yes/no, eight states, prior turns on at the eighth");
    System.out.println("question: " + REFUND.text());
    System.out.println("Each state is 2 prefills. The prior is estimated from these answers");
    System.out.println("and applied from the eighth state of this question on this Jev.");
    System.out.println("The last two lines repeat earlier states. Compare p(Yes).");
    System.out.println("The printed answer is the label with the higher probability.");
    double[] yesProbability = new double[REFUND_STATES.size()];
    for (int index = 0; index < REFUND_STATES.size(); index++) {
      yesProbability[index] = printRefundState(jev, index, REFUND_STATES.get(index));
    }
    for (String state : PRIOR_REPEATS) {
      printRefundRepeat(jev, state, yesProbability[REFUND_STATES.indexOf(state)]);
    }
  }

  private static double printRefundState(final Jev jev, final int index, final String state) {
    int total = REFUND_STATES.size();
    System.out.println("reading state " + (index + 1) + "/" + total + "…");

    JevDecision decision = jev.decide(state, REFUND);
    System.out.println("  " + (index + 1) + "/" + total
      + "  prior=" + priorMark(decision)
      + "  answer=" + decision.answer()
      + "  p(Yes)=" + format(decision.probability("Yes"))
      + "  " + state);
    return decision.probability("Yes");
  }

  private static void printRefundRepeat(final Jev jev, final String state,
                                        final double earlierYes) {
    int earlier = REFUND_STATES.indexOf(state) + 1;
    int total = REFUND_STATES.size();
    System.out.println("reading the same words as " + earlier + "/" + total + "…");

    JevDecision decision = jev.decide(state, REFUND);
    System.out.println("  prior=" + priorMark(decision)
      + "  answer=" + decision.answer()
      + "  p(Yes)=" + format(decision.probability("Yes"))
      + "  was " + format(earlierYes) + " prior=off at " + earlier + "/" + total
      + "  " + state);
  }

  private static String priorMark(final JevDecision decision) {
    return decision.priorApplied() ? "on" : "off";
  }

  private static void printDecision(final JevDecision decision) {
    String name = decision.name().isEmpty() ? "(unnamed)" : decision.name();
    System.out.println(name + " → " + decision.answer()
      + "  level=" + decision.level()
      + "  rotations=" + decision.rotations()
      + "  prior=" + priorMark(decision)
      + "  orderFlip=" + format(decision.orderFlipRate())
      + "  labelMass=" + format(decision.labelMass()));
    for (JevOption option : decision.distribution()) {
      System.out.println("  " + option.option() + "  p=" + format(option.probability()));
    }
  }

  private static void showRequire(final JevDecision decision) {
    try {
      decision.require(JevLevel.L0);
      System.out.println("  require(L0) accepted");
    } catch (IllegalStateException rejected) {
      System.out.println("  require(L0) refused: " + rejected.getMessage());
    }
  }

  private static void heading(final String title) {
    System.out.println();
    System.out.println("--- " + title);
  }

  private static Path checkpoint(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return Path.of(args[0]).toAbsolutePath().normalize();
    }
    return BundledModels.requireChatDemo();
  }

  private static String format(final double value) {
    return String.format(Locale.ROOT, "%.4f", value);
  }
}
