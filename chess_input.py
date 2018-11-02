import numpy as np

class Repr1:

    def __init__(self):
        self.SIZE_PER_COLOR = 48 + 5 * 64
        self.SIZE = 2 * self.SIZE_PER_COLOR + 5
        self.eval_buf = np.ndarray((self.SIZE,))


    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square


    def board_to_array(self, b):
        self.eval_buf[:] = 0

        if not b.turn:
            b = b.mirror()

        for piece_type in range(1, 7):
            balance = 0
            squares = b.pieces(piece_type, True)
            for sq in squares:
                self.eval_buf[self.get_offset(sq, piece_type)] = 1
                balance += 1
            squares = b.pieces(piece_type, False)
            for sq in squares:
                self.eval_buf[self.get_offset(sq, piece_type) + self.SIZE_PER_COLOR] = 1
                balance -= 1
            if piece_type < 6:
                self.eval_buf[self.SIZE - piece_type] = balance

        return self.eval_buf

class Repr2:

    def __init__(self):
        self.SIZE = 48 + 5 * 64 + 1
        self.eval_buf = np.ndarray((self.SIZE,))


    def get_offset(self, square, piece_type):
        if piece_type == 1:
            return square - 8
        else:
            return 48 + (piece_type - 2) * 64 + square


    def board_to_array(self, b):
        self.eval_buf[:] = 0

        for piece_type in range(1, 7):
            squares = b.pieces(piece_type, True)
            for sq in squares:
                self.eval_buf[self.get_offset(sq, piece_type)] = 1
            squares = b.pieces(piece_type, False)
            for sq in squares:
                self.eval_buf[self.get_offset(sq, piece_type)] = -1
        if b.turn:
            self.eval_buf[self.SIZE - 1] = 1
        else:
            self.eval_buf[self.SIZE - 1] = -1

        return self.eval_buf
