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

while True:
    print(board)

    input = repr.board_to_array(board)
    predictions = model.predict([input.reshape(1, repr.SIZE)])
    from_pred = predictions[0].flatten()
    to_pred   = predictions[1].flatten()

    max_prob = 0
    best_move = None

    if board.turn:
        for move in board.generate_legal_moves():
            xor = 0
            if not board.turn:
                xor = 0x38
            prob = from_pred[move.from_square ^ xor] + to_pred[move.to_square ^ xor]
            if best_move is None or prob > max_prob:
                max_prob = prob
                best_move = move
    else:
        best_move = black_searcher.select_move(board)
    print("=> {}".format(board.san(best_move)))
    board.push(best_move)
