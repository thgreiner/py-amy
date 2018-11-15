import sys
import random
import os
import chess
from chess import Board, Move, Piece
import chess.pgn

from chess_input import Repr2D

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
from keras.models import load_model

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 5_000_000
OPENING = 8
MODEL_NAME='model-2d.h5'


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


def phasing(label, moves_in_game, current_move):
    # return label * (1.0 + moves_in_game - current_move) ** -0.8
    return label * (0.97 ** (moves_in_game - current_move))


def create_model():
    board_input = keras.layers.Input(shape = (8, 8, 12), name='board_input')
    castle_input =  keras.layers.Input(shape = (4,), name='castle_input')

    conv1 = keras.layers.Conv2D(96, (3, 3), name='conv1')(board_input)
    norm1 = keras.layers.BatchNormalization(axis = 3, name='norm1')(conv1)
    leaky1 = keras.layers.LeakyReLU(0.01, name='leaky1')(norm1)

    conv2 = keras.layers.Conv2D(64, (3, 3), name='conv2')(leaky1)
    norm2 = keras.layers.BatchNormalization(axis = 3, name='norm2')(conv2)
    leaky2 = keras.layers.LeakyReLU(0.01, name='leaky2')(norm2)

    flatten = keras.layers.Flatten(name='flatten')(leaky2)
    concat = keras.layers.concatenate(inputs=[flatten, castle_input], name='concat')

    dense1 = keras.layers.Dense(256, name='dense1')(concat)
    leaky3 = keras.layers.LeakyReLU(0.01, name='leaky3')(dense1)

    output = keras.layers.Dense(1, activation='tanh', name='output')(leaky3)

    return keras.Model(inputs = (board_input, castle_input), outputs=output)


def parse_pgn_to_training_data(file_name):
    pgn = open(file_name)

    npos = POSITIONS_TO_LEARN_APRIORI
    repr = Repr2D()
    train_data_pos = np.zeros((npos, 8, 8, 12), np.int8)
    train_data_castle = np.zeros((npos, 4), np.int8)
    train_labels = np.zeros((npos), np.float32)

    i = 0
    ngames = 0
    while True:

        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError or ValueError:
            continue

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
            if nmoves >= OPENING and not is_capture:
                train_data_pos[i] = repr.board_to_array(b)
                train_data_castle[i] = repr.castling_to_array(b)
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

    train_data_pos = train_data_pos[:i]
    train_data_castle = train_data_castle[:i]
    train_labels = train_labels[:i]

    return ([train_data_pos, train_data_castle], train_labels)


def train_model_from_pgn(file_name):

    if True:
        model = create_model()
    else:
        model = load_model(MODEL_NAME)

    model.summary()

    optimizer = keras.optimizers.Adadelta()

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'])

    train_data, train_labels = parse_pgn_to_training_data(file_name)

    while True:
        history = model.fit(train_data, train_labels, batch_size=2048, epochs=3,
                             validation_split = 0.1)
        model.save(MODEL_NAME)


if __name__ == '__main__':
    train_model_from_pgn(sys.argv[1])
