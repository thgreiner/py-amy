import unittest

from py_amy.engine.board import Board
from py_amy.engine.constants import *

class BoardTests(unittest.TestCase):
    def testInitialPosition(self):
        board = Board()
        piece, color = board.get(8)
        self.assertEqual(Piece.PAWN, piece)
        self.assertEqual(Color.WHITE, color)

if __name__ == '__main__':
    unittest.main()
