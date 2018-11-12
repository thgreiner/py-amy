import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr2D
import sys

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 10_000_000
OPENING = 5
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
    return 10 * label * (1.0 + moves_in_game - current_move) ** -0.8

def train_model_from_pgn(file_name):
    if False:
        model2 = keras.Sequential([
            keras.layers.Conv2D(30, (3, 3), activation='relu', input_shape=(8, 8, 12)),
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(20, (3, 3), activation='relu'),
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(1)
        ])
    else:
        model2 = load_model(MODEL_NAME)

    model2.summary()


    opt1 = tf.train.AdamOptimizer()

    model2.compile(optimizer=opt1,
                   loss='mean_squared_error',
                   metrics=['mae'])

    pgn = open(file_name)

    npos = POSITIONS_TO_LEARN_APRIORI
    repr = Repr2D()
    train_data = np.zeros((npos, 8, 8, 12))
    train_labels = np.zeros((npos))

    i = 0
    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError or ValueError:
            pass
        if game is None:
            break
        label = label_for_result(game.headers["Result"])
        # if label == 0:
        #     continue
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
        print(i, end='\r')

    print(i)

    npos = i
    train_data = train_data[:npos]
    train_labels = train_labels[:npos]

    while True:
        history = model2.fit(train_data, train_labels, batch_size=1024, epochs=10)
        model2.save(MODEL_NAME)

if __name__ == '__main__':
    train_model_from_pgn(sys.argv[1])
