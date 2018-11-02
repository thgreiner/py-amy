import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr1, Repr2


repr = Repr1()

OPENING = 1

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model2 = load_model("model2.h5")
model2.summary()
opt1 = tf.train.AdamOptimizer()

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



offset = 0
white_searcher = Searcher(lambda board: evaluate(board, model2))
black_searcher = AmySearcher()

while True:

    b = Board()
    # b.set_fen(start_pos)

    game = chess.pgn.Game()
    game.headers["Event"] = "Test Game"
    game.headers["White"] = white_searcher.name
    game.headers["Black"] = black_searcher.name
    node = game

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

        node = node.add_variation(move)  
        print(b.san(move))
        b.push(move)
        print(b)
        print(b.fen())

    game.headers["Result"] = b.result()

    with open("LearnGames.pgn", "a") as f:
        print(game, file=f, end="\n\n")

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

    history = model2.fit(train_data, train_labels, batch_size=1024, epochs=50)
    print(history.history)

    offset += n
    white_searcher, black_searcher = black_searcher, white_searcher

