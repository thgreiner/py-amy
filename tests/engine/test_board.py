import unittest

from py_amy.engine.board import Board
from py_amy.engine.constants import *

class BoardTests(unittest.TestCase):
    def testInitialPosition(self):
        board = Board()
        piece, color = board.get(8)
        self.assertEqual(Piece.PAWN, piece)
        self.assertEqual(Color.WHITE, color)

    def testFromEPD(self):
        board = Board("K1k5/8/8/8/8/8/8/8/8 w - -")
        piece, color = board.get(0)
        self.assertEqual(Piece.KING, piece)
        self.assertEqual(Color.WHITE, color)
        piece, color = board.get(2)
        self.assertEqual(Piece.KING, piece)
        self.assertEqual(Color.BLACK, color)


if __name__ == '__main__':
    unittest.main()
