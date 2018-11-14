import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr2D
import piece_square_eval
from datetime import date

repr = Repr2D()

OPENING = 1

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model2 = load_model("model-2d.h5")

def evaluate(board, model):
    input = repr.board_to_array(board)
    prediction = model.predict([input.reshape(1, 8, 8, 12)]).flatten()
    return prediction[0]


white_searcher = Searcher(lambda board: evaluate(board, model2))
black_searcher = Searcher(lambda board: piece_square_eval.evaluate(board), "PieceSquareTables")
# black_searcher = AmySearcher()
# black_searcher = white_searcher

while True:

    b = Board()
    # b.set_fen(start_pos)

    game = chess.pgn.Game()
    game.headers["Event"] = "Test Game"
    game.headers["White"] = white_searcher.name
    game.headers["Black"] = black_searcher.name
    game.headers["Date"] = date.today().strftime("%Y.%m.%d")
    node = game

    # opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5"
    # opening = "d4 d5"
    opening = "e4 c5 Nf3 Nc6"
    for move in opening.split(" "):
        m = b.parse_san(move)
        node = node.add_variation(m)
        b.push(m)

    while not b.is_game_over():
        print(b)
        print(b.fen())

        if b.turn:
            move = white_searcher.select_move(b)
        else:
            move = black_searcher.select_move(b)

        node = node.add_variation(move)
        print(b.san(move))
        b.push(move)

    game.headers["Result"] = b.result()

    with open("LearnGames.pgn", "a") as f:
        print(game, file=f, end="\n\n")

    print("{} after {} moves.".format(b.result(), b.fullmove_number))

    white_searcher, black_searcher = black_searcher, white_searcher
