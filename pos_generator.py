from chess import Board, Piece
import random

def generate_kqk():
    is_ok = False
    board = Board()
    while not is_ok:
        board.clear()
        positions = random.sample(range(64), 3)
        board.set_piece_at(positions[0], Piece.from_symbol("K"))
        board.set_piece_at(positions[1], Piece.from_symbol("Q"))
        board.set_piece_at(positions[2], Piece.from_symbol("k"))
        is_ok = board.is_valid()
    return board
