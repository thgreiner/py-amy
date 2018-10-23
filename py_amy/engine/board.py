from py_amy.engine.constants import *

INITIAL_POSITON = 'RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w KQkq -'

class Board:
    """
    Represents a chess board.
    """

    def __init__(self, epd = INITIAL_POSITON):
        self.pieces = [
            Piece.KING, None, Piece.KING, None, None, None, None, None,
            Piece.PAWN
        ]
        self.colors = [
            Color.WHITE, None, Color.BLACK, None, None, None, None, None,
            Color.WHITE
        ]
        pass

    def get(self, idx):
        return (self.pieces[idx], self.colors[idx])
