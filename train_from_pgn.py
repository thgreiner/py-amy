import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
from searcher import Searcher, AmySearcher
from chess_input import Repr2D
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import sys

repr = Repr2D()
LEAK=0.1

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 400_000


def label_for_result(result, turn):
    if result == '1-0':
        if turn:
            return [ 1 , 0 ]
        else:
            return [ 0, 1 ]
    if result == '0-1':
        if turn:
            return [ 0, 1 ]
        else:
            return [ 1, 0 ]

    return [ 0.5, 0.5 ]


def phasing(label, moves_in_game, current_move):
    # return label * (1.0 + moves_in_game - current_move) ** -0.8
    return label # * (0.99 ** (moves_in_game - current_move))


def my_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def residual_block(y):
    shortcut = y
    dim_int = 32
    dim_out = 64

    y = keras.layers.Conv2D(dim_int, (1, 1), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)

    y = keras.layers.Conv2D(dim_int, (3, 3), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)

    y = keras.layers.Conv2D(dim_out, (1, 1), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.add([y, shortcut])
    y = keras.layers.LeakyReLU()(y)

    return y

def residual_block2(y):
    shortcut = y
    dim_int = 64
    dim_out = 64

    y = keras.layers.Conv2D(dim_int, (1, 1), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)

    y = keras.layers.SeparableConv2D(dim_int, (3, 3), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    y = keras.layers.add([y, shortcut])
    y = keras.layers.LeakyReLU()(y)

    return y

def create_model():
    board_input = keras.layers.Input(shape = (8, 8, 17), name='board_input')

    dim = 64

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same')(board_input)
    temp = keras.layers.BatchNormalization()(temp)
    temp = keras.layers.LeakyReLU()(temp)

    for i in range(13):
        temp = residual_block2(temp)
        # temp = keras.layers.Conv2D(dim, (3, 3), padding='same')(temp)
        # temp = keras.layers.BatchNormalization()(temp)
        # temp = keras.layers.LeakyReLU()(temp)

    t2 = keras.layers.Conv2D(dim, (3, 3), padding='same')(temp)
    t2 = keras.layers.BatchNormalization()(t2)
    t2 = keras.layers.LeakyReLU()(t2)
    t2 = keras.layers.Conv2D(64, (3, 3), activation='linear', padding='same')(t2)
    move_output = keras.layers.Flatten(name='moves')(t2)

    avg_pooled = keras.layers.GlobalAveragePooling2D()(temp)
    
    temp = keras.layers.Conv2D(5, (1, 1), padding='same')(temp)
    temp = keras.layers.BatchNormalization()(temp)
    temp = keras.layers.LeakyReLU()(temp)
    
    temp = keras.layers.Flatten()(temp)
    temp = keras.layers.concatenate([temp, avg_pooled])

    temp = keras.layers.Dense(256)(temp)
    temp = keras.layers.LeakyReLU()(temp)

    score_output = keras.layers.Dense(2, activation='softmax', name='score')(temp)

    return keras.Model(inputs = board_input, outputs=[move_output, score_output])


if True:
    model = create_model()
else:
    model = load_model("combined-model.h5", custom_objects={'my_categorical_crossentropy': my_categorical_crossentropy})

model.summary()

# opt1 = tf.train.AdamOptimizer()
opt1 = keras.optimizers.Adam()

model.compile(optimizer=opt1,
               loss={'moves': my_categorical_crossentropy, 'score': 'mean_squared_error' },
               metrics=['accuracy', 'mae'])

pgn = open(sys.argv[1])
# pgn = open("LearnGames.pgn")

# Training batch size
BATCH_SIZE = 256

# Checkpoint every x batches
CHECKPOINT = 100

train_data = np.zeros(((BATCH_SIZE, 8, 8, 17)), np.int8)
train_labels1 = np.zeros((BATCH_SIZE, 4096), np.int8)
train_labels2 = np.zeros((BATCH_SIZE, 2), np.float32)
cnt1 = 0

ngames = 0

samples = 0
checkpoint_no = 0
checkpoint_next = CHECKPOINT * BATCH_SIZE

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

    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))

    try:
        for move in game.main_line():
            san = b.san(move)

            piece = b.piece_at(move.from_square).piece_type
            train_data[cnt1] = repr.board_to_array(b)
            train_labels1[cnt1] = repr.move_to_array(b, piece, move)
            train_labels2[cnt1] = label_for_result(result, b.turn)
            cnt1 += 1
            
            if cnt1 == BATCH_SIZE:
                train_labels = [ train_labels1, train_labels2 ]
                results = model.train_on_batch(train_data, train_labels)
                samples += cnt1
                print(samples, results)
                cnt1 = 0
                if samples >= checkpoint_next:
                    checkpoint_no += 1
                    checkpoint_name = "checkpoint-{}.h5".format(checkpoint_no)
                    print("Checkpointing model to {}".format(checkpoint_name))
                    model.save(checkpoint_name)
                    checkpoint_next += CHECKPOINT * BATCH_SIZE

            b.push(move)
            nmoves += 1
    except AttributeError:
        print("Oops - bad game encountered. Skipping it...")
    if cnt1 >= POSITIONS_TO_LEARN_APRIORI:
        break
    print("{}: {}".format(ngames, cnt1), end='\r')

# Train on the remainder of the dataset
train_labels = [ train_labels1[:cnt1], train_labels2[:cnt1] ]
results = model.train_on_batch(train_data[:cnt1], train_labels)
samples += cnt1
print(samples, results)

model.save("move-model.h5")
