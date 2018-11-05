import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 900000

SIZE_PER_COLOR = 49 + 5 * 65
SIZE = 2 * SIZE_PER_COLOR + 3

OPENING = -1

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model1 = keras.Sequential([
    keras.layers.Dense(200, input_shape=(SIZE, )),
#    keras.layers.Dropout(0.2),
    keras.layers.Dense(90),
#    keras.layers.Dropout(0.1),
    keras.layers.Dense(1)
])

if False:
    model2 = keras.Sequential([
        keras.layers.Dense(512, input_shape=(SIZE, )),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1)
    ])
else:
    model2 = load_model("model2.h5")


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


def get_offset(square, piece_type):
    if piece_type == 1:
        return square - 8
    else:
        return 49 + (piece_type - 2) * 65 + square


eval_buf = np.ndarray((SIZE,))


def board_to_array(b):
    global eval_buf
    eval_buf[:] = 0

    total_pieces = 0
    for piece_type in range(1, 7):
        balance = 0
        squares = b.pieces(piece_type, True)
        for sq in squares:
            eval_buf[get_offset(sq, piece_type)] = 1
            balance += 1
            total_pieces += 1
        squares = b.pieces(piece_type, False)
        for sq in squares:
            eval_buf[get_offset(sq, piece_type) + SIZE_PER_COLOR] = 1
            balance -= 1
            total_pieces += 1
        eval_buf[get_offset(64, piece_type)] = balance
    if b.turn:
        eval_buf[SIZE - 2] = 1
    else:
        eval_buf[SIZE - 1] = 1
    eval_buf[SIZE - 3] = total_pieces
    return eval_buf


def evaluate(board, model):
    input = board_to_array(board)
    score = model.predict([input.reshape(1, SIZE)])[0][0]
    if not b.turn:
        score = -score
    score += random.uniform(-0.005, 0.005)
    return score


nodes = 0


def gen_kqk():
    while True:
        pos = random.sample(list(range(64)), 3)
        b = Board()
        b.clear()
        b.set_piece_at(pos[0], Piece.from_symbol('K'))
        b.set_piece_at(pos[1], Piece.from_symbol(random.choice(['Q', 'R'])))
        b.set_piece_at(pos[2], Piece.from_symbol('k'))

        if b.status() == chess.STATUS_VALID:
            return b

pgn = open("ClassicGames.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data = np.zeros((npos, SIZE))
train_labels = np.zeros((npos))

i = 0
while True:
    game = chess.pgn.read_game(pgn)
    label = label_for_result(game.headers["Result"])
    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))
    for move in game.main_line():
        is_capture = b.is_capture(move)
        if nmoves > OPENING and not is_capture:
            train_data[i] = board_to_array(b)
            train_labels[i] = phasing(label, moves_in_game, nmoves)
            i += 1
            train_data[i] = board_to_array(b.mirror())
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

offset = npos
searcher = Searcher(lambda board: evaluate(board, model2))
amy_searcher = AmySearcher()

while True:

    # model1.fit(train_data, train_labels, batch_size=128, epochs=30)
    model2.fit(train_data, train_labels, batch_size=128, epochs=10)

    model2.save("model2.h5")

    # b = Board()
    start_pos = gen_kqk().fen()

    for white_searcher in [searcher, amy_searcher]:
        b = Board()
        b.set_fen(start_pos)

        while not b.is_game_over() and len(b.move_stack) < 30:
            if len(b.move_stack) > OPENING:
                if b.turn:
                    move = white_searcher.select_move(b)
                else:
                    move = searcher.select_move(b)
            else:
                move = random.choice(list(b.generate_legal_moves()))

            print(b.san(move))
            b.push(move)
            print(b)
            print(b.fen())

        print("{} after {} moves.".format(b.result(), b.fullmove_number))

        result = label_for_result(b.result())
        print(result)

        n = 2 * (len(b.move_stack) - OPENING)

        if offset == 0:
            train_data = np.zeros((n, SIZE))
            train_labels = np.zeros((n))
        else:
            new_train_data = np.zeros((n + offset, SIZE))
            new_train_labels = np.zeros((n + offset))
            new_train_data[0:offset] = train_data
            new_train_labels[0:offset] = train_labels
            train_data = new_train_data
            train_labels = new_train_labels

        moves_in_game = len(b.move_stack)
        while len(b.move_stack) > OPENING:
            try:
                m = b.pop()
                i = 2 * (len(b.move_stack) - OPENING)
                train_data[offset + i] = board_to_array(b)
                train_labels[offset + i] = phasing(result, moves_in_game, len(b.move_stack))
                i += 1
                train_data[offset + i] = board_to_array(b.mirror())
                train_labels[offset + i] = -phasing(result, moves_in_game, len(b.move_stack))
            except:
                break

        offset += n
