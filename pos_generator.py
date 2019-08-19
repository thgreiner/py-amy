from chess import Board, Piece
import random

def generate_kxk():
    board = Board()
    while True:
        board.clear()
        positions = random.sample(range(64), 3)
        board.set_piece_at(positions[0], Piece.from_symbol("K"))
        board.set_piece_at(positions[1], Piece.from_symbol(random.choice(['Q', 'R', 'P'])))
        # board.set_piece_at(positions[1], Piece.from_symbol('Q'))
        board.set_piece_at(positions[2], Piece.from_symbol("k"))
        if board.is_valid():
            return board

def generate_kqk():
    board = Board()
    while True:
        board.clear()
        positions = random.sample(range(64), 3)
        bk_pos = positions[2]
        if (bk_pos & 7) in [0, 7] or (bk_pos >> 3) in [0, 7]:
            wk_pos = positions[0]
            dist = abs((bk_pos & 7) - (wk_pos & 7)) + abs((bk_pos >> 3) - (wk_pos >> 3))
            if dist == 2:
                board.set_piece_at(positions[0], Piece.from_symbol("K"))
                board.set_piece_at(positions[1], Piece.from_symbol('Q'))
                board.set_piece_at(positions[2], Piece.from_symbol("k"))
                if board.is_valid():
                    return board

def generate_krk():
    board = Board()
    while True:
        board.clear()
        positions = random.sample(range(64), 3)
        bk_pos = positions[2]
        if (bk_pos & 7) in [0, 7] or (bk_pos >> 3) in [0, 7]:
            wk_pos = positions[0]
            dist = abs((bk_pos & 7) - (wk_pos & 7)) + abs((bk_pos >> 3) - (wk_pos >> 3))
            if dist == 2:
                board.set_piece_at(positions[0], Piece.from_symbol("K"))
                board.set_piece_at(positions[1], Piece.from_symbol('R'))
                board.set_piece_at(positions[2], Piece.from_symbol("k"))
                if board.is_valid():
                    return board
