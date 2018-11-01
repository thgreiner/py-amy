import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr1, Repr2
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0, 1.5])
    plt.show()

repr = Repr1()

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 200

OPENING = 1

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model1 = keras.Sequential([
    keras.layers.Dense(200, input_shape=(repr.SIZE, )),
#    keras.layers.Dropout(0.2),
    keras.layers.Dense(90),
#    keras.layers.Dropout(0.1),
    keras.layers.Dense(1)
])

if True:
    model2 = keras.Sequential([
        keras.layers.Dense(100, input_shape=(repr.SIZE, )),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(100),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(100),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(100),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1)
    ])
else:
    model2 = load_model("model2.h5")

model2.summary()


opt1 = tf.train.AdamOptimizer()
opt2 = tf.train.RMSPropOptimizer(0.001)

model1.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

model2.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


def phasing(label, moves_in_game, current_move):
    return 10 * label * (1.0 + moves_in_game - current_move) ** -0.8


def evaluate(board, model):
    if b.turn:
        input = repr.board_to_array(board)
    else:
        input = repr.board_to_array(board.mirror())

    score = model.predict([input.reshape(1, repr.SIZE)])[0][0]
    # score += random.uniform(-0.005, 0.005)
    return score


pgn = open("ClassicGames.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data = np.zeros((npos, repr.SIZE))
train_labels = np.zeros((npos))

i = 0
while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    label = label_for_result(game.headers["Result"])
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
train_data = np.resize(train_data, (npos, repr.SIZE))
train_labels = np.resize(train_labels, (npos, ))

offset = npos
white_searcher = Searcher(lambda board: evaluate(board, model2))
black_searcher = AmySearcher()

while True:

    # model1.fit(train_data, train_labels, batch_size=128, epochs=30)
    history = model2.fit(train_data, train_labels, batch_size=1024, epochs=50)
    print(history.history)
    # plot_history(history)

    model2.save("model2.h5")

    # b = Board()
    # start_pos = gen_kqk().fen()

    white_searcher, black_searcher = black_searcher, white_searcher

    b = Board()
    # b.set_fen(start_pos)

    opening = [ Move.from_uci("d2d4"), Move.from_uci("d7d5") ]
    while not b.is_game_over():
        if len(b.move_stack) > OPENING:
            if b.turn:
                move = white_searcher.select_move(b)
            else:
                move = black_searcher.select_move(b)
        else:
            # move = random.choice(list(b.generate_legal_moves()))
            move = opening[len(b.move_stack)]

        print(b.san(move))
        b.push(move)
        print(b)
        print(b.fen())

    print("{} after {} moves.".format(b.result(), b.fullmove_number))

    result = label_for_result(b.result())
    print(result)

    n = len(b.move_stack) - OPENING

    if offset == 0:
        train_data = np.zeros((n, repr.SIZE))
        train_labels = np.zeros((n))
    else:
        new_train_data = np.zeros((n + offset, repr.SIZE))
        new_train_labels = np.zeros((n + offset))
        new_train_data[0:offset] = train_data
        new_train_labels[0:offset] = train_labels
        train_data = new_train_data
        train_labels = new_train_labels

    moves_in_game = len(b.move_stack)
    while len(b.move_stack) > OPENING:
        try:
            m = b.pop()
            i = (len(b.move_stack) - OPENING)
            train_data[offset + i] = repr.board_to_array(b)
            if b.turn:
                train_labels[i] = phasing(label, moves_in_game, nmoves)
            else:
                train_labels[i] = -phasing(label, moves_in_game, nmoves)
        except:
            break

    offset += n
