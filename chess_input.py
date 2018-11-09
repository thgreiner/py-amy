import numpy as np
from chess import Board, Piece

white_pieces = [ None, 'P', 'N', 'B', 'R', 'Q', 'K' ]
black_pieces = [ None, 'p', 'n', 'b', 'r', 'q', 'k' ]
THRESHOLD = 0.4

class Repr1:

    def __init__(self):
        self.SIZE_PER_COLOR = 48 + 5 * 64
        self.SIZE = 2 * self.SIZE_PER_COLOR + 9
        self.eval_buf = np.ndarray((self.SIZE,))


    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square


    def board_to_array(self, b):
        buf = np.zeros(self.SIZE, np.int8)

        if not b.turn:
            b = b.mirror()

        for piece_type in range(1, 7):
            balance = 0
            squares = b.pieces(piece_type, True)
            for sq in squares:
                buf[self.get_offset(sq, piece_type)] = 1
                balance += 1
            squares = b.pieces(piece_type, False)
            for sq in squares:
                buf[self.get_offset(sq, piece_type) + self.SIZE_PER_COLOR] = 1
                balance -= 1
            if piece_type < 6:
                buf[self.SIZE - piece_type] = balance

        if b.has_kingside_castling_rights(True):
            buf[self.SIZE - 6] = 1
        if b.has_queenside_castling_rights(True):
            buf[self.SIZE - 7] = 1
        if b.has_queenside_castling_rights(False):
            buf[self.SIZE - 8] = 1
        if b.has_queenside_castling_rights(False):
            buf[self.SIZE - 9] = 1

        return buf

    def array_to_board(self, a):
        b = Board()
        b.clear()
        for piece_type in range(1, 7):
            if piece_type == 1:
                r = range(8, 56)
            else:
                r = range(0, 64)
            for sq in r:
                if a[self.get_offset(sq, piece_type)] >= THRESHOLD:
                    b.set_piece_at(sq, Piece.from_symbol(white_pieces[piece_type]))
                if a[self.get_offset(sq, piece_type) + self.SIZE_PER_COLOR] >= THRESHOLD:
                    b.set_piece_at(sq, Piece.from_symbol(black_pieces[piece_type]))
        return b

class Repr2:

    def __init__(self):
        self.SIZE = 48 + 5 * 64 + 9


    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square


    def board_to_array(self, b):
        buf = np.zeros(self.SIZE, np.int8)

        if not b.turn:
            b = b.mirror()

        for piece_type in range(1, 7):
            squares = b.pieces(piece_type, True)
            balance = 0
            for sq in squares:
                buf[self.get_offset(sq, piece_type)] = 1
                balance += 1
            squares = b.pieces(piece_type, False)
            for sq in squares:
                buf[self.get_offset(sq, piece_type)] = -1
                balance -= 1
            if piece_type < 6:
                buf[self.SIZE - piece_type] = balance

        if b.has_kingside_castling_rights(True):
            buf[self.SIZE - 6] = 1
        if b.has_queenside_castling_rights(True):
            buf[self.SIZE - 7] = 1
        if b.has_queenside_castling_rights(False):
            buf[self.SIZE - 8] = 1
        if b.has_queenside_castling_rights(False):
            buf[self.SIZE - 9] = 1

        return buf

weights = [ None, 1, 1, 1, 1, 1, 1]
class Repr3:

    def __init__(self):
        self.SIZE_PER_COLOR = 48 + 5 * 64
        self.SIZE = 2 * self.SIZE_PER_COLOR
        self.eval_buf = np.ndarray((self.SIZE,))


    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square


    def board_to_array(self, b):
        buf = np.zeros(self.SIZE)

        if not b.turn:
            b = b.mirror()

        for piece_type in range(1, 7):
            squares = b.pieces(piece_type, True)
            for sq in squares:
                buf[self.get_offset(sq, piece_type)] = weights[piece_type]
            squares = b.pieces(piece_type, False)
            for sq in squares:
                buf[self.get_offset(sq, piece_type) + self.SIZE_PER_COLOR] = weights[piece_type]
        return buf

    def array_to_board(self, a):
        b = Board()
        b.clear()
        for piece_type in range(1, 7):
            if piece_type == 1:
                r = range(8, 56)
            else:
                r = range(0, 64)
            for sq in r:
                if a[self.get_offset(sq, piece_type)] >= weights[piece_type] * .5:
                    b.set_piece_at(sq, Piece.from_symbol(white_pieces[piece_type]))
                if a[self.get_offset(sq, piece_type) + self.SIZE_PER_COLOR] >= weights[piece_type] * .5:
                    b.set_piece_at(sq, Piece.from_symbol(black_pieces[piece_type]))
        return b


class BoardAndMoveRepr:

    def __init__(self):
        self.SIZE_PER_COLOR = 48 + 5 * 64
        self.SIZE = 2 * self.SIZE_PER_COLOR + 4

    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square

    def board_to_array(self, b):
        buf = np.zeros(self.SIZE, np.int8)
        xor = 0

        if not b.turn:
            xor = 0x38

        for piece_type in range(1, 7):
            squares = b.pieces(piece_type, b.turn)
            for sq in squares:
                buf[self.get_offset(sq ^ xor, piece_type)] = 1
            squares = b.pieces(piece_type, not b.turn)
            for sq in squares:
                buf[self.get_offset(sq ^ xor, piece_type) + self.SIZE_PER_COLOR] = 1

        offset = 2 * self.SIZE_PER_COLOR
        if b.has_kingside_castling_rights(b.turn):
            buf[offset] = 1
        offset += 1
        if b.has_queenside_castling_rights(b.turn):
            buf[offset] = 1
        offset += 1
        if b.has_queenside_castling_rights(not b.turn):
            buf[offset] = 1
        offset += 1
        if b.has_queenside_castling_rights(not b.turn):
            buf[offset] = 1
        return buf

    def move_to_array(self, b, move):
        buf1 = np.zeros(64, np.int8)
        buf2 = np.zeros(64, np.int8)

        xor = 0

        if not b.turn:
            xor = 0x38

        buf1[move.from_square ^ xor] = 1
        buf2[move.to_square ^ xor] = 1

        return (buf1, buf2)
