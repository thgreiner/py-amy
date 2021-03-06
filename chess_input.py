import numpy as np
import chess
from chess import Board, Piece
from scipy.sparse import csr_matrix


class Repr2D:
    def __init__(self):
        queen_dirs = [-1, 1, -8, 8, -7, 7, -9, 9]
        knight_dirs = [-15, 15, -17, 17, -10, 10, -6, 6]
        pawn_dirs = [7, 8, 9]
        self.queen_indexes = dict()
        self.knight_indexes = dict()
        self.underpromo_indexes = {
            chess.KNIGHT: dict(),
            chess.BISHOP: dict(),
            chess.ROOK: dict(),
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
            for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                self.underpromo_indexes[piece][delta] = idx
                idx += 1

        self.num_planes = 19

        # print("Generated {} indexes for moves.".format(idx))

    def _is_knight_move(self, move):
        file_dist = abs((move.from_square & 7) - (move.to_square & 7))
        rank_dist = abs((move.from_square >> 3) - (move.to_square >> 3))
        return (file_dist == 1 and rank_dist == 2) or (
            file_dist == 2 and rank_dist == 1
        )

    def plane_index(self, move, xor):
        delta = (move.to_square ^ xor) - (move.from_square ^ xor)
        if move.promotion and move.promotion != chess.QUEEN:
            return self.underpromo_indexes[move.promotion][delta]
        elif self._is_knight_move(move):
            return self.knight_indexes[delta]
        else:
            return self.queen_indexes[delta]

    def _coords(self, sq):
        return (sq >> 3, sq & 7)

    def _store_board(self, b, buf, turn, plane_offset):
        xor = 0 if turn else 0x38

        for piece_type in range(1, 7):
            offset = plane_offset + 2 * (piece_type - 1)
            squares = b.pieces(piece_type, turn)
            for sq in squares:
                rank, file = self._coords(sq ^ xor)
                buf[rank, file, offset] = 1
            squares = b.pieces(piece_type, not turn)
            for sq in squares:
                rank, file = self._coords(sq ^ xor)
                buf[rank, file, offset + 1] = 1

    def board_to_array(self, board):
        buf = np.zeros((8, 8, self.num_planes), np.int8)

        turn = board.turn

        self._store_board(board, buf, turn, 0)

        offset = 12

        if board.ep_square:
            xor = 0 if turn else 0x38
            rank, file = self._coords(board.ep_square ^ xor)
            buf[rank, file, offset] = 1

        if board.has_kingside_castling_rights(board.turn):
            buf[:, :, offset + 1] = 1
        if board.has_queenside_castling_rights(board.turn):
            buf[:, :, offset + 2] = 1
        if board.has_kingside_castling_rights(not board.turn):
            buf[:, :, offset + 3] = 1
        if board.has_queenside_castling_rights(not board.turn):
            buf[:, :, offset + 4] = 1

        # One plane just ones so the network can detect the board edge
        buf[:, :, offset + 5] = 1

        buf[:, :, offset + 6] = board.halfmove_clock / 100.0

        return buf

    def legal_moves_mask(self, b):
        buf = np.zeros((64, 73), np.int8)

        xor = 0 if b.turn else 0x38

        for move in b.generate_legal_moves():
            buf[move.to_square ^ xor, self.plane_index(move, xor)] = 1

        return buf.flatten()

    def move_to_array(self, b, move):
        buf = np.zeros((64, 73), np.int8)
        xor = 0 if b.turn else 0x38
        buf[move.to_square ^ xor, self.plane_index(move, xor)] = 1
        return buf.reshape(4672)

    def policy_to_array(self, b, policy):
        # buf = np.zeros((64, 7), np.int8)
        buf = np.zeros((64, 73), np.float32)

        xor = 0 if b.turn else 0x38

        for san, value in policy.items():
            move = b.parse_san(san)
            # buf[move.from_square ^ xor][0] = 1
            # buf[move.to_square ^ xor][piece] = 1
            buf[move.to_square ^ xor, self.plane_index(move, xor)] = value

        # return buf.reshape(8, 8, 7)
        buf = buf / np.sum(buf)
        return csr_matrix(buf)
