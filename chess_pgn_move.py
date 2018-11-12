import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import BoardAndMoveRepr
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

repr = BoardAndMoveRepr()

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 200_000


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


if True:
    inputs = keras.layers.Input(shape = (repr.SIZE,))
    hidden1 = keras.layers.Dense(
        512,
        # activity_regularizer = keras.regularizers.l1(.001),
        name = 'hidden1')(inputs)
    leaky1  = keras.layers.LeakyReLU(name='leaky1')(hidden1)
    hidden2 = keras.layers.Dense(
        512,
        # activity_regularizer = keras.regularizers.l1(.001),
        name='hidden2')(leaky1)
    leaky2  = keras.layers.LeakyReLU(name='leaky2')(hidden2)
    hidden3 = keras.layers.Dense(
        512,
        name='hidden3')(leaky2)
    leaky3  = keras.layers.LeakyReLU(name='leaky3')(hidden3)

    output1 = keras.layers.Dense(64, activation='relu')(leaky3)
    outputp = keras.layers.Dense(64, activation='relu')(leaky3)
    outputn = keras.layers.Dense(64, activation='relu')(leaky3)
    outputb = keras.layers.Dense(64, activation='relu')(leaky3)
    outputr = keras.layers.Dense(64, activation='relu')(leaky3)
    outputq = keras.layers.Dense(64, activation='relu')(leaky3)
    outputk = keras.layers.Dense(64, activation='relu')(leaky3)
    output3 = keras.layers.Dense(1)(leaky3)

    model1 = keras.Model(inputs = inputs, outputs=(
        output1, 
        outputp, outputn, outputb, outputr, outputq, outputk))
    model2 = keras.Model(inputs = inputs, outputs=(
        output3))

else:
    model1 = load_model("move-model.h5")
    model2 = load_model("score-model.h5")

model1.summary()

# opt1 = tf.train.AdamOptimizer()
opt1 = keras.optimizers.Adadelta()

model1.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

model2.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

pgn = open("PGN/ClassicGames.pgn")
# pgn = open("LearnGames.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data_1 = np.zeros((npos, repr.SIZE), np.int8)
train_labels1 = np.zeros((npos, 64), np.int8)
train_labelsp = np.zeros((npos, 64), np.int8)
train_labelsn = np.zeros((npos, 64), np.int8)
train_labelsb = np.zeros((npos, 64), np.int8)
train_labelsr = np.zeros((npos, 64), np.int8)
train_labelsq = np.zeros((npos, 64), np.int8)
train_labelsk = np.zeros((npos, 64), np.int8)
cnt1 = 0

train_data_2 = np.zeros((npos, repr.SIZE), np.int8)
train_labels3 = np.zeros((npos, ), np.int8)
cnt2 = 0

i = 0
ngames = 0

while True:
    try:
        game = chess.pgn.read_game(pgn)
    except UnicodeDecodeError or ValueError:
        pass
    if game is None:
        break

    ngames += 1
    # if label == 0:
    #     continue
    result = game.headers["Result"]
    result_label = label_for_result(result)

    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))
    for move in game.main_line():
        if (b.turn and result_label == 1) or (not b.turn and result_label == -1):
            piece = b.piece_at(move.from_square).piece_type
            train_data_1[cnt1] = repr.board_to_array(b)
            train_labels1[cnt1], \
            train_labelsp[cnt1], \
            train_labelsn[cnt1], \
            train_labelsb[cnt1], \
            train_labelsr[cnt1], \
            train_labelsq[cnt1], \
            train_labelsk[cnt1] = repr.move_to_array(b, piece, move)
            cnt1 += 1

        if not b.is_capture(move):
            train_data_2[cnt2] = repr.board_to_array(b)
            if b.turn:
                train_labels3[cnt2] = result_label
            else:
                train_labels3[cnt2] = -result_label
            cnt2 += 1

        b.push(move)
        if cnt2 >= npos:
            break
    if cnt2 >= npos:
        break
    print("{}: {} {}".format(ngames, cnt1, cnt2), end='\r')

print("{} {}".format(cnt1, cnt2))

if cnt1 < npos:
    train_data_1 = train_data_1[:cnt1][:]
    train_labels1 = train_labels1[:cnt1][:]
    train_labelsp = train_labelsp[:cnt1][:]
    train_labelsn = train_labelsn[:cnt1][:]
    train_labelsb = train_labelsb[:cnt1][:]
    train_labelsr = train_labelsr[:cnt1][:]
    train_labelsq = train_labelsq[:cnt1][:]
    train_labelsk = train_labelsk[:cnt1][:]

if cnt2 < npos:
    train_data_2 = train_data_2[:cnt2][:]
    train_labels3 = train_labels3[:cnt2][:]

while True:
    history = model1.fit(train_data_1, (
        train_labels1, 
        train_labelsp,
        train_labelsn,
        train_labelsb,
        train_labelsr,
        train_labelsq,
        train_labelsk), batch_size=1024, epochs=5)
    model1.save("move-model.h5")

    history = model2.fit(train_data_2, train_labels3, batch_size=1024, epochs=5)
    model2.save("score-model.h5")
