import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr2D
import sys

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 4_000_000
OPENING = 8
MODEL_NAME='model-2d.h5'

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


def phasing(label, moves_in_game, current_move):
    # return label * (1.0 + moves_in_game - current_move) ** -0.8
    return label * (0.97 ** (moves_in_game - current_move))

def train_model_from_pgn(file_name):
    if False:
        model2 = keras.Sequential([
            keras.layers.Conv2D(128, (3, 3), input_shape=(8, 8, 12)),
            keras.layers.BatchNormalization(axis = 3),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(96, (3, 3)),
            keras.layers.BatchNormalization(axis = 3),
            keras.layers.LeakyReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(1, activation='tanh')
        ])
    else:
        model2 = load_model(MODEL_NAME)

    model2.summary()


    opt1 = keras.optimizers.Adadelta()

    model2.compile(optimizer=opt1,
                   loss='mean_squared_error',
                   metrics=['mae'])

    pgn = open(file_name)

    npos = POSITIONS_TO_LEARN_APRIORI
    repr = Repr2D()
    train_data = np.zeros((npos, 8, 8, 12), np.int8)
    train_labels = np.zeros((npos), np.float32)

    i = 0
    ngames = 0
    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError or ValueError:
            pass
        if game is None:
            break
        label = label_for_result(game.headers["Result"])
        if label == 0:
            continue
        b = game.board()
        nmoves = 0
        moves_in_game = len(list(game.main_line()))
        for move in game.main_line():
            is_capture = b.is_capture(move)
            if nmoves > OPENING and not is_capture:
                train_data[i] = repr.board_to_array(b)
                if b.turn:
                    train_labels[i] = phasing(label, moves_in_game, nmoves)
                else:
                    train_labels[i] = -phasing(label, moves_in_game, nmoves)
                i += 1
            b.push(move)
            nmoves += 1
            if i >= npos:
                break
        if i >= npos:
            break
        ngames += 1
        print("Games: {}, Positions: {}".format(ngames, i), end='\r')

    print("Games: {}, Positions: {}".format(ngames, i))

    npos = i
    train_data = train_data[:npos]
    train_labels = train_labels[:npos]

    while True:
        history = model2.fit(train_data, train_labels, batch_size=128, epochs=3)
        model2.save(MODEL_NAME)


if __name__ == '__main__':
    train_model_from_pgn(sys.argv[1])
