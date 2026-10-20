package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld.Board;
import com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld.Move;
import com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld.Piece;
import com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld.Side;
import com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld.Sq;
import java.util.List;
import org.junit.jupiter.api.Test;

class RussianCheckersHelloWorldTest {

  private static List<String> notations(final Board board, final Side side) {
    return board.legalMoves(side).stream().map(Move::notation).toList();
  }

  private static Move only(final Board board, final Side side) {
    List<Move> moves = board.legalMoves(side);
    assertEquals(1, moves.size());
    return moves.getFirst();
  }

  private static Move move(final Board board, final String notation) {
    return board.legalMoves(Side.WHITE).stream()
      .filter(item -> item.notation().equals(notation))
      .findFirst()
      .orElseThrow();
  }

  private static int count(final Board board, final Piece piece) {
    int found = 0;
    for (int rank = 0; rank < 8; rank++) {
      for (int file = 0; file < 8; file++) {
        if (board.pieceAt(new Sq(file, rank)).filter(piece::equals).isPresent()) {
          found++;
        }
      }
    }
    return found;
  }

  private static Board board(final String text) {
    Board board = Board.empty();
    for (String token : text.split("\\s+")) {
      Piece piece = switch (token.charAt(token.length() - 1)) {
        case 'w' -> Piece.WHITE_MAN;
        case 'W' -> Piece.WHITE_KING;
        case 'b' -> Piece.BLACK_MAN;
        case 'B' -> Piece.BLACK_KING;
        default -> throw new IllegalArgumentException(token);
      };
      board = board.place(Sq.parse(token.substring(0, token.length() - 1)), piece);
    }
    return board;
  }

  @Test
  void openingOffersSevenWhiteManSteps() {
    Board board = Board.initial();
    assertEquals(12, count(board, Piece.WHITE_MAN));
    assertEquals(12, count(board, Piece.BLACK_MAN));
    assertEquals(
      List.of("a3-b4", "c3-b4", "c3-d4", "e3-d4", "e3-f4", "g3-f4", "g3-h4"),
      notations(board, Side.WHITE));
  }

  @Test
  void manStepsForwardOnlyAndDoesNotCaptureItsOwnColour() {
    assertEquals(List.of("c3-b4", "c3-d4"), notations(board("c3w"), Side.WHITE));
    assertEquals(List.of("b2-a3", "c3-b4", "c3-d4"), notations(board("c3w b2w"), Side.WHITE));
    assertEquals(List.of("b6-a5", "b6-c5"), notations(board("b6b"), Side.BLACK));
  }

  @Test
  void captureIsMandatoryAndAManJumpsBackward() {
    assertEquals(List.of("c3xa1"), notations(board("c3w e3w b2b"), Side.WHITE));
  }

  @Test
  void manCaptureChainRemovesVictimsOnlyWhenTheMoveEnds() {
    Board board = board("c3w d4b f6b");
    assertEquals(List.of("c3xe5xg7"), notations(board, Side.WHITE));

    Board after = board.play(only(board, Side.WHITE));
    assertEquals(Piece.WHITE_MAN, after.pieceAt(Sq.parse("g7")).orElseThrow());
    assertTrue(after.pieceAt(Sq.parse("c3")).isEmpty());
    assertTrue(after.pieceAt(Sq.parse("d4")).isEmpty());
    assertTrue(after.pieceAt(Sq.parse("f6")).isEmpty());
  }

  @Test
  void quietArrivalOnTheLastRankCrownsTheMan() {
    Board board = board("a7w");
    Board after = board.play(only(board, Side.WHITE));
    assertEquals("a7-b8", only(board, Side.WHITE).notation());
    assertEquals(Piece.WHITE_KING, after.pieceAt(Sq.parse("b8")).orElseThrow());
  }

  @Test
  void manCrownedMidCaptureContinuesAsAKing() {
    Board board = board("b6w c7b e7b");
    assertEquals(List.of("b6xd8xf6", "b6xd8xg5", "b6xd8xh4"), notations(board, Side.WHITE));

    Move landed = move(board, "b6xd8xf6");
    Board after = board.play(landed);
    assertEquals(Piece.WHITE_KING, after.pieceAt(Sq.parse("f6")).orElseThrow());
    assertTrue(after.pieceAt(Sq.parse("c7")).isEmpty());
    assertTrue(after.pieceAt(Sq.parse("e7")).isEmpty());
  }

  @Test
  void kingSlidesAnyDistanceAndChoosesWhereToLandACapture() {
    assertTrue(notations(board("d4W"), Side.WHITE).containsAll(
      List.of("d4-a1", "d4-a7", "d4-g1", "d4-h8")));
    assertFalse(notations(board("d4W"), Side.WHITE).contains("d4-d5"));

    assertEquals(List.of("a1xe5", "a1xf6", "a1xg7", "a1xh8"),
      notations(board("a1W d4b"), Side.WHITE));
  }

  @Test
  void capturedPieceStaysUntilTheSequenceEndsAndBlocksASecondJump() {
    assertEquals(List.of("d4xa1", "d4xg7", "d4xh8"), notations(board("d4W f6b b2b"), Side.WHITE));
  }

  @Test
  void blackViewPutsThePlayersMenOnTheBottomRank() {
    List<Sq> whiteView = RussianCheckersHelloWorld.displaySquares(false);
    List<Sq> blackView = RussianCheckersHelloWorld.displaySquares(true);

    assertEquals(Sq.parse("a8"), whiteView.getFirst());
    assertEquals(Sq.parse("h1"), whiteView.getLast());
    assertEquals(Sq.parse("h1"), blackView.getFirst());
    assertEquals(List.of("h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"),
      blackView.subList(56, 64).stream().map(Sq::label).toList());
  }

  @Test
  void oneLegalMoveIsOfferedForEnter() {
    List<Move> only = board("c3w d4b f6b").legalMoves(Side.WHITE);
    assertEquals("Your move [c3xe5xg7, Enter]: ", RussianCheckersHelloWorld.movePrompt(only));
    assertEquals("c3xe5xg7",
      RussianCheckersHelloWorld.blankLineMove(only).orElseThrow().notation());

    List<Move> several = board("c3w").legalMoves(Side.WHITE);
    assertEquals("Your move: ", RussianCheckersHelloWorld.movePrompt(several));
    assertTrue(RussianCheckersHelloWorld.blankLineMove(several).isEmpty());
  }

  @Test
  void coordinatesAcceptColumnRowPairsAndACapturePath() {
    assertEquals(List.of(Sq.parse("a3"), Sq.parse("b4")),
      RussianCheckersHelloWorld.squaresIn("A 3 B 4"));
    assertEquals(
      List.of(Sq.parse("c3"), Sq.parse("e5"), Sq.parse("g7")),
      RussianCheckersHelloWorld.squaresIn("c3xe5-g7"));

    Board board = board("c3w d4b f6b");
    List<Move> legal = board.legalMoves(Side.WHITE);
    assertEquals("c3xe5xg7",
      RussianCheckersHelloWorld.matching(legal, "c3 e5").getFirst().notation());
    assertEquals("c3xe5xg7",
      RussianCheckersHelloWorld.matching(legal, "C 3 E 5 G 7").getFirst().notation());
    assertTrue(RussianCheckersHelloWorld.matching(legal, "a3 b4").isEmpty());

    Board branch = board("b6w c7b e7b");
    assertEquals(3,
      RussianCheckersHelloWorld.matching(branch.legalMoves(Side.WHITE), "B 6 D 8").size());
  }
}
