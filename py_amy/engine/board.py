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

def iterateSingleRow(row, d):
    column_count = 0
    for s in row:
        if s >= '1' and s <= '8':
            for i in range(int(s)):
                column_count += 1
                yield None
        else:
            column_count += 1
            yield d[s]
    while column_count < 8:
        column_count += 1
        yield None

def iterateRows(rows, d):
    for row in rows:
        yield from iterateSingleRow(row, d)

class Board:
    """
    Represents a chess board.
    """

    def __init__(self, epd = INITIAL_POSITON):
        t = epd.split(' ')
        pcs = t[0].split('/')
        while len(pcs) < 8:
            pcs.append('8')
        pcs.reverse()
        self.pieces = list(iterateRows(pcs, PIECE_BY_NAME))
        self.colors = list(iterateRows(pcs, COLOR_BY_NAME))

    def get(self, idx):
        return (self.pieces[idx], self.colors[idx])
