import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher
from chess_input import BoardAndMoveRepr
import piece_square_eval

repr = BoardAndMoveRepr()

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model("move-model.h5")
black_searcher = Searcher(lambda board: piece_square_eval.evaluate(board))

board = Board()

def sort_key(from_pred, to_pred, board, move, xor):
    type = board.piece_at(move.from_square).piece_type
    fr = move.from_square ^ xor
    to = move.to_square ^ xor
    return -from_pred[fr] * to_pred[type-1][to]

while not board.is_game_over():
    print(board)

    input = repr.board_to_array(board)
    predictions = model.predict([input.reshape(1, repr.SIZE)])
    from_pred = predictions[0].flatten()
    to_pred   = [ predictions[i].flatten() for i in range(1,7)]
    print("Prediction: {}".format(predictions[7].flatten()[0]))

    max_prob = 0
    best_move = None

    if board.turn:
        xor = 0
        if not board.turn:
            xor = 0x38
        moves = list(board.generate_legal_moves())
        moves = sorted(moves, key=lambda m: sort_key(from_pred, to_pred, board, m, xor))
        print(moves)
        best_move = moves[0]
    else:
        best_move = black_searcher.select_move(board)
    print("=> {}".format(board.san(best_move)))
    board.push(best_move)
