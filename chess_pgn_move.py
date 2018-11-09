import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import BoardAndMoveRepr
#import matplotlib.pyplot as plt

#def plot_history(history):
#    plt.figure()
#    plt.xlabel('Epoch')
#    plt.ylabel('Mean Abs Error')
#    plt.plot(history.epoch, np.array(history.history['loss']),
#             label='Train Loss')
#    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
#             label = 'Val loss')
#    plt.legend()
#    plt.ylim([0, 1.5])
#    plt.show()

repr = BoardAndMoveRepr()

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 500_000

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

if True:
    inputs = keras.layers.Input(shape = (repr.SIZE,))
    hidden1 = keras.layers.Dense(512, activation='relu')(inputs)
    hidden2 = keras.layers.Dense(512, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(512, activation='relu')(hidden2)

    output1 = keras.layers.Dense(64, activation='softmax')(hidden3)
    output2 = keras.layers.Dense(repr.SIZE2)(hidden3)

    model = keras.Model(inputs = inputs, outputs=(output1, output2))
else:
    model = load_model("move-model.h5")

model.summary()

opt1 = tf.train.AdamOptimizer()

model.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

pgn = open("TWIC.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data = np.zeros((npos, repr.SIZE), np.int8)
train_labels1 = np.zeros((npos, 64), np.int8)
train_labels2 = np.zeros((npos, repr.SIZE2), np.int8)

i = 0

while True:
    try:
        game = chess.pgn.read_game(pgn)
    except UnicodeDecodeError or ValueError:
        pass
    if game is None:
        break

    # if label == 0:
    #     continue
    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))
    for move in game.main_line():
        piece = b.piece_at(move.from_square).piece_type
        train_data[i] = repr.board_to_array(b)
        train_labels1[i], train_labels2[i] = repr.move_to_array(b, piece, move)
        i += 1
        b.push(move)
        if i >= npos:
            break
    if i >= npos:
        break
    print(i, end='\r')

print(i)

if i < npos:
    npos = i
    train_data = np.resize(train_data, (npos, repr.SIZE))
    train_labels1 = np.resize(train_labels1, (npos, 64))
    train_labels2 = np.resize(train_labels2, (npos, repr.SIZE2))

while True:
    history = model.fit(train_data, (train_labels1, train_labels2), batch_size=1024, epochs=10)
    # plot_history(history)
    model.save("move-model.h5")
