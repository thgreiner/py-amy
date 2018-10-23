from py_amy.engine.constants import *

INITIAL_POSITON = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'

PIECE_BY_NAME = {
    'p': Piece.PAWN,
    'P': Piece.PAWN,
    'n': Piece.KNIGHT,
    'N': Piece.KNIGHT,
    'b': Piece.BISHOP,
    'B': Piece.BISHOP,
    'r': Piece.ROOK,
    'R': Piece.ROOK,
    'q': Piece.QUEEN,
    'Q': Piece.QUEEN,
    'k': Piece.KING,
    'K': Piece.KING,
}

COLOR_BY_NAME = {
    'p': Color.BLACK,
    'P': Color.WHITE,
    'n': Color.BLACK,
    'N': Color.WHITE,
    'b': Color.BLACK,
    'B': Color.WHITE,
    'r': Color.BLACK,
    'R': Color.WHITE,
    'q': Color.BLACK,
    'Q': Color.WHITE,
    'k': Color.BLACK,
    'K': Color.WHITE,
}

def piecesFor(c):
    for s in c:
        if s >= '1' and s <= '8':
            for i in range(int(s)):
                yield None
        else:
            yield PIECE_BY_NAME[s]

def colorsFor(c):
    for s in c:
        if s >= '1' and s <= '8':
            for i in range(int(s)):
                yield None
        else:
            yield COLOR_BY_NAME[s]

class Board:
    """
    Represents a chess board.
    """

    def __init__(self, epd = INITIAL_POSITON):
        t = epd.split(' ')
        pcs = t[0].split('/')
        pcs.reverse()
        self.pieces = list(piecesFor("".join(pcs)))
        self.colors = list(colorsFor("".join(pcs)))

    def get(self, idx):
        return (self.pieces[idx], self.colors[idx])
