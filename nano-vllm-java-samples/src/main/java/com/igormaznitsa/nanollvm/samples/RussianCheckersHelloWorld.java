package com.igormaznitsa.nanollvm.samples;

import static java.util.Comparator.comparing;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toMap;

import com.igormaznitsa.nanollvm.jev.Jev;
import com.igormaznitsa.nanollvm.jev.JevDecision;
import com.igormaznitsa.nanollvm.jev.JevLevel;
import com.igormaznitsa.nanollvm.jev.JevQuestion;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Terminal Russian draughts. You pick your color once; Qwen Jeff plays the other. Jeff chooses a
 * legal move with one typed decision ({@link JevLevel#RAW}) and the sample prints how long that
 * decision took.
 *
 * <p>Moves are coordinates, in the same shape as a terminal checkers board: {@code a3 b4},
 * {@code a3-b4}, or {@code A 3 B 4}. A capture lists every landing ({@code c3 e5 g7}).
 *
 * <p>The Jeff calls are the part to copy into another program. Read them in this order:
 * <ol>
 * <li>{@link #playWithJeff} opens one decision engine on a causal chat model.</li>
 * <li>{@link #positionText} is the situation the model reads. It is not the list of answers.</li>
 * <li>{@link #choose} skips the model when the rules leave only one move, and times the rest.</li>
 * <li>{@link #narrow} keeps each question inside 26 options.</li>
 * <li>{@link #ask} is the only call to {@link Jev#decide}.</li>
 * </ol>
 * The model does not write a move. This program lists the legal moves, and the decision is which
 * of those strings is most likely. {@link JevHelloWorld} is the wider tour: a choice, a yes/no,
 * a score, and the label prior.
 *
 * <p>Args: optional chat checkpoint (default {@code models/Jeff-Qwen3.5-0.8B}). From the
 * repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld}
 *
 * @since 1.5.0
 */
public final class RussianCheckersHelloWorld {

  /**
   * Upper bound of {@link JevQuestion#choice}. The model answers with one letter, {@code A}
   * through {@code Z}.
   */
  private static final int CHOICE_LIMIT = 26;
  private static final int MOVE_LIMIT = 240;
  private static final int REPETITION_DRAW = 3;
  private static final Pattern SQUARE = Pattern.compile("([a-hA-H])\\s*([1-8])");
  private static final String RESET = "\u001B[0m";
  private static final String WHITE_PAINT = "\u001B[97m";
  private static final String BLACK_PAINT = "\u001B[31m";
  private static final String DIM = "\u001B[2m";

  private final BufferedReader in;
  private final boolean color;
  private final Seat white;
  private final Seat black;
  private final Jev jev;

  private RussianCheckersHelloWorld(
    final BufferedReader in,
    final boolean color,
    final Seat white,
    final Seat black,
    final Jev jev
  ) {
    this.in = requireNonNull(in, "in");
    this.color = color;
    this.white = requireNonNull(white, "white");
    this.black = requireNonNull(black, "black");
    this.jev = requireNonNull(jev, "jev");
  }

  /**
   * Asks which color you play, gives the other color to Qwen Jeff, then plays.
   */
  public static void main(final String[] args) throws IOException {
    explain();
    try (BufferedReader in = new BufferedReader(
      new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      Side yours = askYourColour(in);
      Seat white = yours == Side.WHITE ? Seat.YOU : Seat.JEFF;
      Seat black = yours == Side.BLACK ? Seat.YOU : Seat.JEFF;
      System.out.println();
      System.out.println(
        "You play " + yours.label() + ". Qwen Jeff plays " + yours.opposite().label() + ".");
      playWithJeff(args, in, white, black);
    }
  }

  /**
   * Loads a causal chat model and opens the engine Jeff plays through.
   *
   * <p>{@link LLM.Builder#toJev()} builds that engine from the model. Closing the {@link Jev}
   * closes the engine and leaves the model open, so both belong in one {@code try}.
   * {@link JevLevel#RAW} is one prefill and keeps the options in the order given.
   * {@link JevLevel#L0} reads every rotation of an unordered choice, one prefill per option.
   * Use L0 when the order of the labels must not sway the answer. RAW is the level used here,
   * because a move already has many candidates and each extra rotation would be another full
   * read. Any causal chat checkpoint can replace the default Jeff-Qwen3.5-0.8B.
   */
  private static void playWithJeff(
    final String[] args,
    final BufferedReader in,
    final Seat white,
    final Seat black
  ) throws IOException {
    Path path = checkpoint(args);
    System.out.println("Loading " + path);
    System.out.println(
      "Jeff reads the legal moves once (RAW). The time under a move is that readout.");
    try (LlmModel model = LlmModelFactory.make(path);
         Jev jev = LLM.builder(model).jevLevel(JevLevel.RAW).toJev()) {
      System.out.println("architecture=" + model.architectureName());
      new RussianCheckersHelloWorld(in, useColor(), white, black, jev).play();
    }
  }

  private static void explain() {
    System.out.println("Russian checkers. White moves first. Your pieces sit at the bottom.");
    System.out.println(
      "A man steps one square diagonally forward. A king (O* or X*) moves any distance.");
    System.out.println(
      "A capture is required. Men jump one square either way. Kings jump any distance.");
    System.out.println(
      "Enter a move as a3 b4, a3-b4, or A 3 B 4. List every landing of a long capture.");
    System.out.println("If only one move is legal, press Enter to play it. q or 0 quits.");
    System.out.println();
  }

  private static Side askYourColour(final BufferedReader in) throws IOException {
    System.out.println("Your colour? Qwen Jeff plays the other.");
    System.out.println("  1  White (you move first)");
    System.out.println("  2  Black");
    while (true) {
      System.out.print("Choice [1-2, Enter=1]: ");
      System.out.flush();
      String line = in.readLine();
      if (line == null || line.isBlank() || line.strip().equals("1")) {
        return Side.WHITE;
      }
      if (line.strip().equals("2")) {
        return Side.BLACK;
      }
      System.out.println("Enter 1 or 2.");
    }
  }

  private static Path checkpoint(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return Path.of(args[0]).toAbsolutePath().normalize();
    }
    return BundledModels.require(BundledModels.JEFF_QWEN35_0_8B);
  }

  private static boolean useColor() {
    return System.getenv("NO_COLOR") == null
      && !"false".equalsIgnoreCase(NanoLlvmProps.systemProperty(NanoLlvmProps.PROP_COLOR));
  }

  private static String seatName(final Seat seat) {
    return seat == Seat.YOU ? "you" : "Qwen Jeff";
  }

  static List<Sq> displaySquares(final boolean yoursAtBottom) {
    int rank = yoursAtBottom ? 0 : 7;
    int rankStep = yoursAtBottom ? 1 : -1;
    int fileStart = yoursAtBottom ? 7 : 0;
    int fileStep = yoursAtBottom ? -1 : 1;
    List<Sq> squares = new ArrayList<>();
    for (int rowIndex = 0; rowIndex < 8; rowIndex++) {
      for (int column = 0; column < 8; column++) {
        squares.add(new Sq(fileStart + column * fileStep, rank));
      }
      rank += rankStep;
    }
    return squares;
  }

  private static String fileLabels(final List<Sq> bottomRow) {
    StringBuilder labels = new StringBuilder("     ");
    for (int column = 0; column < bottomRow.size(); column++) {
      if (column > 0) {
        labels.append("   ");
      }
      labels.append((char) ('a' + bottomRow.get(column).file()));
    }
    return labels.toString();
  }

  /**
   * Situation text, the first argument of {@link Jev#decide}.
   *
   * <p>Put here the facts the choice depends on: whose turn it is, the rules that bind the
   * answer, and the board. Leave the candidate moves out. Those strings are the question's
   * options, and the model answers with a letter that stands for one of them. The question words
   * can stay the same on every turn. A new option list is a new {@link JevQuestion}.
   */
  private static String positionText(final Board board, final Side side) {
    return "Russian checkers. White (o man, O king) moves toward row 8. "
      + "Black (x man, X king) moves toward row 1. "
      + "Men step one diagonal square forward and capture one square either way. "
      + "Kings move and capture any distance on a diagonal. Take a capture when one exists.\n"
      + board.diagram()
      + "Files a to h from left to right. " + side.label() + " to move.";
  }

  private static boolean isQuit(final String line) {
    String key = line.toLowerCase(Locale.ROOT);
    return key.equals("q") || key.equals("quit") || key.equals("0") || key.equals("exit");
  }

  static List<Sq> squaresIn(final String line) {
    requireNonNull(line, "line");
    Matcher matcher = SQUARE.matcher(line);
    List<Sq> squares = new ArrayList<>();
    while (matcher.find()) {
      int file = matcher.group(1).toLowerCase(Locale.ROOT).charAt(0) - 'a';
      int rank = matcher.group(2).charAt(0) - '1';
      squares.add(new Sq(file, rank));
    }
    return List.copyOf(squares);
  }

  static String movePrompt(final List<Move> legal) {
    requireNonNull(legal, "legal");
    if (legal.size() != 1) {
      return "Your move: ";
    }
    return "Your move [" + legal.getFirst().notation() + ", Enter]: ";
  }

  static Optional<Move> blankLineMove(final List<Move> legal) {
    requireNonNull(legal, "legal");
    return legal.size() == 1 ? Optional.of(legal.getFirst()) : Optional.empty();
  }

  static List<Move> matching(final List<Move> legal, final String line) {
    requireNonNull(legal, "legal");
    List<Sq> asked = squaresIn(line);
    if (asked.size() < 2) {
      return List.of();
    }
    return legal.stream()
      .filter(move -> move.path().size() >= asked.size()
        && move.path().subList(0, asked.size()).equals(asked))
      .toList();
  }

  private void play() throws IOException {
    Board board = Board.initial();
    Side side = Side.WHITE;
    Map<String, Integer> seen = new HashMap<>();
    for (int ply = 1; ply <= MOVE_LIMIT; ply++) {
      List<Move> legal = board.legalMoves(side);
      this.printBoard(board, side, legal);
      if (legal.isEmpty()) {
        System.out.println(side.opposite().label() + " wins. " + side.label() + " has no move.");
        return;
      }
      if (seen.merge(board.fingerprint(side), 1, Integer::sum) >= REPETITION_DRAW) {
        System.out.println("Draw — this position has come up three times.");
        return;
      }

      if (this.seat(side) == Seat.YOU) {
        Optional<Move> chosen = this.readHuman(legal);
        if (chosen.isEmpty()) {
          System.out.println("Stopped.");
          return;
        }
        board = board.play(chosen.get());
        System.out.println(this.playerName(side) + " plays " + chosen.get().notation());
      } else {
        if (legal.size() > 1) {
          System.out.println("Qwen Jeff is choosing...");
        }
        Played played = this.choose(board, side, legal);
        board = board.play(played.move());
        this.printPlayed(side, played);
      }
      System.out.println();
      side = side.opposite();
    }
    System.out.println("Draw — " + MOVE_LIMIT + " moves and neither side has won.");
  }

  private void printBoard(final Board board, final Side side, final List<Move> legal) {
    List<Sq> squares = displaySquares(this.seat(Side.BLACK) == Seat.YOU);
    String bar = "   +---+---+---+---+---+---+---+---+";
    System.out.println(bar);
    for (int rowIndex = 0; rowIndex < 8; rowIndex++) {
      List<Sq> rowSquares = squares.subList(rowIndex * 8, rowIndex * 8 + 8);
      StringBuilder row = new StringBuilder();
      row.append(' ').append(rowSquares.getFirst().rank() + 1).append(" |");
      for (Sq sq : rowSquares) {
        row.append(this.cell(board, sq)).append('|');
      }
      System.out.println(row);
      System.out.println(bar);
    }
    System.out.println(fileLabels(squares.subList(56, 64)));
    System.out.println();
    System.out.println("White: " + seatName(this.white) + "    Black: " + seatName(this.black));
    System.out.println(side.label() + " to move");
    if (!legal.isEmpty()) {
      System.out.println("Legal: " + legal.stream().map(Move::notation).collect(joining("  ")));
    }
  }

  private String cell(final Board board, final Sq sq) {
    return board.pieceAt(sq)
      .map(this::paint)
      .orElseGet(() -> this.emptyCell(sq.dark()));
  }

  private String paint(final Piece piece) {
    String body = piece.king() ? " " + piece.glyph() + "*" : " " + piece.glyph() + " ";
    if (!this.color) {
      return body;
    }
    String tone = piece.side() == Side.WHITE ? WHITE_PAINT : BLACK_PAINT;
    return tone + body + RESET;
  }

  private String emptyCell(final boolean dark) {
    if (!dark) {
      return "   ";
    }
    return this.color ? DIM + " · " + RESET : " · ";
  }

  private Optional<Move> readHuman(final List<Move> legal) throws IOException {
    while (true) {
      System.out.print(movePrompt(legal));
      System.out.flush();
      String line = this.in.readLine();
      if (line == null) {
        return Optional.empty();
      }
      String trimmed = line.strip();
      if (trimmed.isEmpty()) {
        Optional<Move> pressed = blankLineMove(legal);
        if (pressed.isPresent()) {
          return pressed;
        }
        continue;
      }
      if (isQuit(trimmed)) {
        return Optional.empty();
      }

      List<Move> matched = matching(legal, trimmed);
      if (matched.size() == 1) {
        return Optional.of(matched.getFirst());
      }
      if (matched.size() > 1) {
        System.out.println("Several moves start that way:");
        matched.forEach(move -> System.out.println("  " + move.notation()));
        continue;
      }
      System.out.println("Not a legal move. Use a3 b4 or A 3 B 4. A capture must be taken.");
    }
  }

  /**
   * Picks Jeff's move and times only the readout.
   *
   * <p>One legal move is not a decision. {@link JevQuestion#choice} requires at least two
   * options, so that move is returned with no model call and the printed line says
   * {@code only legal move}. The clock starts after that check. The milliseconds are the
   * prefill, not the checkpoint load in {@link #playWithJeff}.
   */
  private Played choose(final Board board, final Side side, final List<Move> legal) {
    if (legal.size() == 1) {
      return Played.onlyMove(legal.getFirst());
    }
    long started = System.nanoTime();
    Choice choice = this.narrow(positionText(board, side), side, legal);
    long millis = (System.nanoTime() - started) / 1_000_000L;
    return new Played(choice.move(), millis, choice.probability(), false);
  }

  /**
   * Brings the legal moves down to one question {@link #ask} can send.
   *
   * <p>A choice holds 2 to {@link #CHOICE_LIMIT} options. A longer list is split into groups of
   * that size. Each group is its own {@link Jev#decide}. A group of one move is kept without a
   * call. The winners are reduced the same way until {@link #CHOICE_LIMIT} or fewer remain, and
   * that last call is the move that is played. Every call sits inside the clock started by
   * {@link #choose}.
   */
  private Choice narrow(final String state, final Side side, final List<Move> moves) {
    List<Move> pool = moves;
    while (pool.size() > CHOICE_LIMIT) {
      List<Move> winners = new ArrayList<>();
      for (int offset = 0; offset < pool.size(); offset += CHOICE_LIMIT) {
        List<Move> chunk = pool.subList(offset, Math.min(offset + CHOICE_LIMIT, pool.size()));
        winners.add(chunk.size() == 1 ? chunk.getFirst() : this.ask(state, side, chunk).move());
      }
      pool = List.copyOf(winners);
    }
    if (pool.size() == 1) {
      return new Choice(pool.getFirst(), OptionalDouble.empty());
    }
    return this.ask(state, side, pool);
  }

  /**
   * One typed choice. This is the call to copy.
   *
   * <p>{@link Jev#decide} does not parse a sentence. The prompt stops where the answer letter
   * would be, and the decision is how likely each letter is. {@link JevQuestion#choice} takes
   * the question text and your own option strings, here the move notations such as
   * {@code c3-d4}. The model is shown {@code A. c3-d4}, {@code B. e3-f4}, and so on.
   * {@link JevDecision#answer()} returns the option string, not the letter.
   * {@link JevDecision#probability(String)} is that option's share of the answer.
   * Map the string back onto the move you put in the list. An answer outside the list means the
   * question and the list no longer describe the same set.
   */
  private Choice ask(final String state, final Side side, final List<Move> moves) {
    JevDecision decision = this.jeff().decide(
      state,
      JevQuestion.choice(
        "Which move should " + side.label() + " play?",
        moves.stream().map(Move::notation).toList()));
    Move move = moves.stream()
      .filter(item -> item.notation().equals(decision.answer()))
      .findFirst()
      .orElseThrow(() -> new IllegalStateException("decision missed the legal moves"));
    return new Choice(move, OptionalDouble.of(decision.probability(decision.answer())));
  }

  /**
   * Prints the move, then what the readout cost. {@code only legal move} means {@link #choose}
   * did not call the model. {@code ms} is the prefill time and {@code p} is
   * {@link JevDecision#probability(String)} for the chosen option. A time with no {@code p} means
   * {@link #narrow} was left with a single survivor and never asked a final question.
   */
  private void printPlayed(final Side side, final Played played) {
    System.out.println(this.playerName(side) + " plays " + played.move().notation());
    if (played.forced()) {
      System.out.println("  only legal move");
      return;
    }
    if (played.probability().isPresent()) {
      System.out.printf(
        Locale.ROOT,
        "  %d ms   p=%.4f%n",
        played.millis(),
        played.probability().getAsDouble());
      return;
    }
    System.out.printf(Locale.ROOT, "  %d ms%n", played.millis());
  }

  private Jev jeff() {
    return requireNonNull(this.jev, "jeff");
  }

  private Seat seat(final Side side) {
    return side == Side.WHITE ? this.white : this.black;
  }

  private String playerName(final Side side) {
    return side.label() + " (" + seatName(this.seat(side)) + ")";
  }

  private enum Seat {
    YOU,
    JEFF
  }

  enum Side {
    WHITE,
    BLACK;

    int forward() {
      return this == WHITE ? 1 : -1;
    }

    int promotionRank() {
      return this == WHITE ? 7 : 0;
    }

    String label() {
      return this == WHITE ? "White" : "Black";
    }

    Side opposite() {
      return this == WHITE ? BLACK : WHITE;
    }
  }

  enum Piece {
    WHITE_MAN(Side.WHITE, false, 'o'),
    WHITE_KING(Side.WHITE, true, 'O'),
    BLACK_MAN(Side.BLACK, false, 'x'),
    BLACK_KING(Side.BLACK, true, 'X');

    private final Side side;
    private final boolean king;
    private final char glyph;

    Piece(final Side side, final boolean king, final char glyph) {
      this.side = side;
      this.king = king;
      this.glyph = glyph;
    }

    Side side() {
      return this.side;
    }

    boolean king() {
      return this.king;
    }

    char glyph() {
      return this.glyph;
    }

    Piece crownedOn(final Sq sq) {
      if (this.king || sq.rank() != this.side.promotionRank()) {
        return this;
      }
      return this.side == Side.WHITE ? WHITE_KING : BLACK_KING;
    }
  }

  private record Played(Move move, long millis, OptionalDouble probability, boolean forced) {

    private static Played onlyMove(final Move move) {
      return new Played(move, 0L, OptionalDouble.empty(), true);
    }
  }

  private record Choice(Move move, OptionalDouble probability) {
  }

  record Sq(int file, int rank) {

    Sq {
      if (file < 0 || file > 7 || rank < 0 || rank > 7) {
        throw new IllegalArgumentException("square off the board");
      }
    }

    static Sq parse(final String label) {
      List<Sq> squares = squaresIn(label);
      if (squares.size() != 1) {
        throw new IllegalArgumentException("not a square: " + label);
      }
      return squares.getFirst();
    }

    String label() {
      return "" + (char) ('a' + this.file) + (this.rank + 1);
    }

    boolean dark() {
      return (this.file + this.rank) % 2 == 0;
    }
  }

  record Move(List<Sq> path, List<Sq> captured, Piece result) {

    Move {
      path = List.copyOf(requireNonNull(path, "path"));
      captured = List.copyOf(requireNonNull(captured, "captured"));
      requireNonNull(result, "result");
      if (path.size() < 2) {
        throw new IllegalArgumentException("a move needs a start and a landing");
      }
      if (captured.isEmpty() && path.size() != 2) {
        throw new IllegalArgumentException("a quiet move is one landing");
      }
      if (!captured.isEmpty() && captured.size() != path.size() - 1) {
        throw new IllegalArgumentException("each capture step needs a victim");
      }
    }

    String notation() {
      String joiner = this.captured.isEmpty() ? "-" : "x";
      return this.path.stream().map(Sq::label).collect(joining(joiner));
    }
  }

  static final class Board {

    private static final List<Step> DIRECTIONS = List.of(
      new Step(1, 1), new Step(1, -1), new Step(-1, 1), new Step(-1, -1));

    private final Piece[][] cells;

    private Board(final Piece[][] cells) {
      this.cells = cells;
    }

    static Board empty() {
      return new Board(new Piece[8][8]);
    }

    static Board initial() {
      Piece[][] cells = new Piece[8][8];
      for (int rank = 0; rank < 3; rank++) {
        fillDark(cells, rank, Piece.WHITE_MAN);
      }
      for (int rank = 5; rank < 8; rank++) {
        fillDark(cells, rank, Piece.BLACK_MAN);
      }
      return new Board(cells);
    }

    private static void fillDark(final Piece[][] cells, final int rank, final Piece piece) {
      for (int file = 0; file < 8; file++) {
        if ((file + rank) % 2 == 0) {
          cells[rank][file] = piece;
        }
      }
    }

    private static boolean onBoard(final int file, final int rank) {
      return file >= 0 && file < 8 && rank >= 0 && rank < 8;
    }

    Board place(final Sq sq, final Piece piece) {
      requireNonNull(sq, "sq");
      requireNonNull(piece, "piece");
      if (!sq.dark()) {
        throw new IllegalArgumentException(sq.label() + " is a light square");
      }
      Piece[][] next = this.copy();
      next[sq.rank()][sq.file()] = piece;
      return new Board(next);
    }

    Optional<Piece> pieceAt(final Sq sq) {
      requireNonNull(sq, "sq");
      return Optional.ofNullable(this.cells[sq.rank()][sq.file()]);
    }

    List<Move> legalMoves(final Side side) {
      requireNonNull(side, "side");
      List<Move> captures = this.captures(side);
      List<Move> found = captures.isEmpty() ? this.quietMoves(side) : captures;
      return found.stream()
        .collect(toMap(Move::notation, move -> move, (left, right) -> left, LinkedHashMap::new))
        .values().stream()
        .sorted(comparing(Move::notation))
        .toList();
    }

    Board play(final Move move) {
      requireNonNull(move, "move");
      Piece[][] next = this.copy();
      Sq from = move.path().getFirst();
      next[from.rank()][from.file()] = null;
      for (Sq captured : move.captured()) {
        next[captured.rank()][captured.file()] = null;
      }
      Sq to = move.path().getLast();
      next[to.rank()][to.file()] = move.result();
      return new Board(next);
    }

    String fingerprint(final Side side) {
      StringBuilder text = new StringBuilder();
      text.append(side.name());
      for (Piece[] rank : this.cells) {
        for (Piece piece : rank) {
          text.append(piece == null ? '.' : piece.glyph());
        }
      }
      return text.toString();
    }

    String diagram() {
      StringBuilder text = new StringBuilder();
      for (int rank = 7; rank >= 0; rank--) {
        text.append(rank + 1).append(' ');
        for (int file = 0; file < 8; file++) {
          Piece piece = this.cells[rank][file];
          text.append(piece == null ? '.' : piece.glyph());
        }
        text.append('\n');
      }
      return text.toString();
    }

    private List<Move> quietMoves(final Side side) {
      List<Move> found = new ArrayList<>();
      this.eachPiece(side, (piece, from) -> {
        if (piece.king()) {
          this.addKingSlides(piece, from, found);
        } else {
          this.addManStep(piece, from, found);
        }
      });
      return found;
    }

    private void addManStep(final Piece piece, final Sq from, final List<Move> out) {
      for (Step step : DIRECTIONS) {
        if (step.rank() != piece.side().forward()) {
          continue;
        }
        int file = from.file() + step.file();
        int rank = from.rank() + step.rank();
        if (!onBoard(file, rank)) {
          continue;
        }
        Sq to = new Sq(file, rank);
        if (!this.isEmpty(to)) {
          continue;
        }
        out.add(new Move(List.of(from, to), List.of(), piece.crownedOn(to)));
      }
    }

    private void addKingSlides(final Piece piece, final Sq from, final List<Move> out) {
      for (Step step : DIRECTIONS) {
        int file = from.file() + step.file();
        int rank = from.rank() + step.rank();
        while (onBoard(file, rank)) {
          Sq to = new Sq(file, rank);
          if (!this.isEmpty(to)) {
            break;
          }
          out.add(new Move(List.of(from, to), List.of(), piece));
          file += step.file();
          rank += step.rank();
        }
      }
    }

    private List<Move> captures(final Side side) {
      List<Move> found = new ArrayList<>();
      this.eachPiece(side, (piece, from) ->
        this.followCaptures(piece, from, List.of(from), List.of(), found));
      return found;
    }

    private void followCaptures(
      final Piece piece,
      final Sq at,
      final List<Sq> path,
      final List<Sq> captured,
      final List<Move> out
    ) {
      List<Hop> hops = this.hops(piece, at, path, captured);
      if (hops.isEmpty()) {
        if (!captured.isEmpty()) {
          out.add(new Move(path, captured, piece));
        }
        return;
      }

      for (Hop hop : hops) {
        Piece arrived = piece.crownedOn(hop.land());
        List<Sq> nextPath = new ArrayList<>(path);
        nextPath.add(hop.land());
        List<Sq> nextCaptured = new ArrayList<>(captured);
        nextCaptured.add(hop.victim());
        this.relocate(at, hop.land(), arrived)
          .followCaptures(arrived, hop.land(), nextPath, nextCaptured, out);
      }
    }

    private List<Hop> hops(
      final Piece piece,
      final Sq at,
      final List<Sq> path,
      final List<Sq> captured
    ) {
      return piece.king()
        ? this.kingHops(piece.side(), at, path, captured)
        : this.manHops(piece.side(), at, path, captured);
    }

    private List<Hop> manHops(
      final Side side,
      final Sq at,
      final List<Sq> path,
      final List<Sq> captured
    ) {
      List<Hop> hops = new ArrayList<>();
      for (Step step : DIRECTIONS) {
        int midFile = at.file() + step.file();
        int midRank = at.rank() + step.rank();
        int landFile = midFile + step.file();
        int landRank = midRank + step.rank();
        if (!onBoard(midFile, midRank) || !onBoard(landFile, landRank)) {
          continue;
        }

        Sq victim = new Sq(midFile, midRank);
        Sq land = new Sq(landFile, landRank);
        if (this.isNotEnemy(victim, side) || captured.contains(victim)) {
          continue;
        }
        if (!this.isEmpty(land) || path.contains(land)) {
          continue;
        }
        hops.add(new Hop(land, victim));
      }
      return hops;
    }

    private List<Hop> kingHops(
      final Side side,
      final Sq at,
      final List<Sq> path,
      final List<Sq> captured
    ) {
      List<Hop> hops = new ArrayList<>();
      for (Step step : DIRECTIONS) {
        int file = at.file() + step.file();
        int rank = at.rank() + step.rank();
        while (onBoard(file, rank) && this.isEmpty(new Sq(file, rank))) {
          file += step.file();
          rank += step.rank();
        }
        if (!onBoard(file, rank)) {
          continue;
        }

        Sq victim = new Sq(file, rank);
        if (this.isNotEnemy(victim, side) || captured.contains(victim)) {
          continue;
        }

        file += step.file();
        rank += step.rank();
        while (onBoard(file, rank)) {
          Sq land = new Sq(file, rank);
          if (!this.isEmpty(land) || path.contains(land)) {
            break;
          }
          hops.add(new Hop(land, victim));
          file += step.file();
          rank += step.rank();
        }
      }
      return hops;
    }

    private void eachPiece(final Side side, final PieceVisitor visitor) {
      for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
          Piece piece = this.cells[rank][file];
          if (piece != null && piece.side() == side) {
            visitor.visit(piece, new Sq(file, rank));
          }
        }
      }
    }

    private Board relocate(final Sq from, final Sq to, final Piece piece) {
      Piece[][] next = this.copy();
      next[from.rank()][from.file()] = null;
      next[to.rank()][to.file()] = piece;
      return new Board(next);
    }

    private boolean isEmpty(final Sq sq) {
      return this.cells[sq.rank()][sq.file()] == null;
    }

    private boolean isNotEnemy(final Sq sq, final Side side) {
      Piece piece = this.cells[sq.rank()][sq.file()];
      return piece == null || piece.side() == side;
    }

    private Piece[][] copy() {
      Piece[][] next = new Piece[8][8];
      IntStream.range(0, 8)
        .forEachOrdered(rank -> System.arraycopy(this.cells[rank], 0, next[rank], 0, 8));
      return next;
    }

    @FunctionalInterface
    private interface PieceVisitor {
      void visit(Piece piece, Sq square);
    }

    private record Step(int file, int rank) {
    }

    private record Hop(Sq land, Sq victim) {
    }
  }
}
