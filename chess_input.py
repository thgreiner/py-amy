import numpy as np
import chess
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

        self.SIZE2 = 6 * 64

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
        if b.has_kingside_castling_rights(not b.turn):
            buf[offset] = 1
        offset += 1
        if b.has_queenside_castling_rights(not b.turn):
            buf[offset] = 1
        return buf

    def move_to_array(self, b, piece, move):
        buf = np.zeros((7, 64), np.int8)

        xor = 0

        if not b.turn:
            xor = 0x38

        buf[0][move.from_square ^ xor] = 1
        buf[piece][move.to_square ^ xor] = 1

        return (buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6])

class Repr2D:
    
    def __init__(self):
        queen_dirs = [ -1, 1, -8, 8, -7, 7, -9, 9 ]
        knight_dirs = [ -15, 15, -17, 17, -10, 10, -6, 6]
        pawn_dirs = [ 7, 8, 9 ]
        self.queen_indexes = dict()
        self.knight_indexes = dict()
        self.underpromo_indexes = {
            chess.KNIGHT: dict(),
            chess.BISHOP: dict(),
            chess.ROOK: dict()
        }

        idx = 0

        for dir in queen_dirs:
            for i in range(1, 8):
                delta = i * dir
                self.queen_indexes[delta] = idx
                idx += 1

        for delta in knight_dirs:
            self.knight_indexes[delta] = idx
            idx += 1

        for delta in pawn_dirs:
            for piece in [ chess.KNIGHT, chess.BISHOP, chess.ROOK ]:
                self.underpromo_indexes[piece][delta] = idx
                idx += 1

        print("Generated {} indexes for moves.".format(idx))


    def is_knight_move(self, move):
        file_dist = abs((move.from_square & 7) - (move.to_square & 7))
        rank_dist = abs((move.from_square >> 3) - (move.to_square >> 3))
        return (file_dist == 1 and rank_dist == 2) or (file_dist == 2 and rank_dist == 1)

    def plane_index(self, move, xor):
        delta = (move.to_square ^ xor) - (move.from_square ^ xor)
        if move.promotion and move.promotion != chess.QUEEN:
            return self.underpromo_indexes[move.promotion][delta]
        elif self.is_knight_move(move):
            return self.knight_indexes[delta]
        else:
            return self.queen_indexes[delta]

        
    def coords(self, sq):
        return (sq >> 3, sq & 7)


    def board_to_array(self, b):
        buf = np.zeros((8, 8, 17), np.int8)
        xor = 0

        if not b.turn:
            xor = 0x38

        for piece_type in range(1, 7):
            offset = 2 * (piece_type - 1)
            squares = b.pieces(piece_type, b.turn)
            for sq in squares:
                rank, file = self.coords(sq ^ xor)
                buf[rank, file, offset] = 1
            squares = b.pieces(piece_type, not b.turn)
            for sq in squares:
                rank, file = self.coords(sq ^ xor)
                buf[rank, file, offset + 1] = 1

        if b.ep_square:
            rank, file = self.coords(b.ep_square ^ xor)
            buf[rank, file, 12] = 1

        if b.has_kingside_castling_rights(b.turn):
            buf[0, 6, 13] = 1
        if b.has_queenside_castling_rights(b.turn):
            buf[0, 2, 13] = 1
        if b.has_kingside_castling_rights(not b.turn):
            buf[7, 6, 14] = 1
        if b.has_queenside_castling_rights(not b.turn):
            buf[7, 2, 14] = 1
        
        # Center
        buf[3, 3, 15] = 1
        buf[3, 4, 15] = 1
        buf[4, 3, 15] = 1
        buf[4, 4, 15] = 1

        for i in range(8):
            buf[i, 0, 16] = 1
            buf[i, 7, 16] = 1
            buf[0, i, 16] = 1
            buf[7, i, 16] = 1
        
        return buf

    def castling_to_array(self, b):
        buf = np.zeros((4), np.int8)
        if b.has_kingside_castling_rights(b.turn):
            buf[0] = 1
        if b.has_queenside_castling_rights(b.turn):
            buf[1] = 1
        if b.has_kingside_castling_rights(not b.turn):
            buf[2] = 1
        if b.has_queenside_castling_rights(not b.turn):
            buf[3] = 1
        return buf

    def move_to_array(self, b, move):
        # buf = np.zeros((64, 7), np.int8)
        buf = np.zeros((64, 73), np.int8)

        xor = 0 if b.turn else 0x38

        # buf[move.from_square ^ xor][0] = 1
        # buf[move.to_square ^ xor][piece] = 1
        buf[move.from_square ^ xor, self.plane_index(move, xor)] = 1
        
        # return buf.reshape(8, 8, 7)
        return buf.reshape(4672)
