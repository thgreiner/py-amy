import unittest

from py_amy.engine.board import Board
from py_amy.engine.constants import *

class BoardTests(unittest.TestCase):
    def testInitialPosition(self):
        board = Board()
        for wp in range(8,16):
            piece, color = board.get(wp)
            self.assertEqual(Piece.PAWN, piece)
            self.assertEqual(Color.WHITE, color)
        for bp in range(48,56):
            piece, color = board.get(bp)
            self.assertEqual(Piece.PAWN, piece)
            self.assertEqual(Color.BLACK, color)
        for empty in range(16, 48):
            piece, color = board.get(empty)
            self.assertEqual(None, piece)
            self.assertEqual(None, color)
        for i in range(0, 8):
            piece, color = board.get(i)
            self.assertEqual(Color.WHITE, color)
            piece, color = board.get(56+i)
            self.assertEqual(Color.BLACK, color)
        for i in [0, 7]:
            piece, color = board.get(i)
            self.assertEqual(Piece.ROOK, piece)
            piece, color = board.get(56+i)
            self.assertEqual(Piece.ROOK, piece)
        for i in [1, 6]:
            piece, color = board.get(i)
            self.assertEqual(Piece.KNIGHT, piece)
            piece, color = board.get(56+i)
            self.assertEqual(Piece.KNIGHT, piece)
        for i in [2, 5]:
            piece, color = board.get(i)
            self.assertEqual(Piece.BISHOP, piece)
            piece, color = board.get(56+i)
            self.assertEqual(Piece.BISHOP, piece)
        for i in [3]:
            piece, color = board.get(i)
            self.assertEqual(Piece.QUEEN, piece)
            piece, color = board.get(56+i)
            self.assertEqual(Piece.QUEEN, piece)
        for i in [4]:
            piece, color = board.get(i)
            self.assertEqual(Piece.KING, piece)
            piece, color = board.get(56+i)
            self.assertEqual(Piece.KING, piece)


    def testFromEPD(self):
        board = Board("8/8/8/8/8/8/8/8/K1k5 w - -")
        piece, color = board.get(0)
        self.assertEqual(Piece.KING, piece)
        self.assertEqual(Color.WHITE, color)
        piece, color = board.get(2)
        self.assertEqual(Piece.KING, piece)
        self.assertEqual(Color.BLACK, color)


if __name__ == '__main__':
    unittest.main()
