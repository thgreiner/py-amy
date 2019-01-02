import chess
from chess import Board
import numpy as np
import chess.pgn
from chess_input import Repr2D
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import sys

# Training batch size
BATCH_SIZE = 2048

# Checkpoint every x batches
CHECKPOINT = 50


def label_for_result(result, turn):
    if result == '1-0':
        if turn:
            return 1
        else:
            return -1
    if result == '0-1':
        if turn:
            return -1
        else:
            return 1

    return 0


def phasing(label, moves_in_game, current_move):
    # return label * (1.0 + moves_in_game - current_move) ** -0.8
    return label # * (0.99 ** (moves_in_game - current_move))


def my_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def residual_block(y, dim):
    shortcut = y
    y = keras.layers.Conv2D(2 * dim, (1, 1), padding='same', activation='elu')(y)
    y = keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='elu')(y)
    y = keras.layers.Conv2D(dim, (1, 1), padding='same', activation='elu')(y)
    y = keras.layers.add([y, shortcut])
    return y


def create_model():
    board_input = keras.layers.Input(shape = (8, 8, 17), name='board_input')

    dim = 80

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same', activation='elu')(board_input)
    for i in range(13):
        temp = residual_block(temp, dim)

    t2 = keras.layers.Conv2D(dim, (3, 3), padding='same', activation='elu')(temp)
    t2 = keras.layers.Conv2D(73, (3, 3), activation='linear', padding='same')(t2)
    move_output = keras.layers.Flatten(name='moves')(t2)

    avg_pooled = keras.layers.GlobalAveragePooling2D()(temp)

    temp = keras.layers.Conv2D(5, (1, 1), padding='same', activation='elu')(temp)
    temp = keras.layers.Flatten()(temp)
    temp = keras.layers.concatenate([temp, avg_pooled])
    temp = keras.layers.Dense(256, activation='elu')(temp)

    score_output = keras.layers.Dense(1, activation='tanh', name='score')(temp)

    return keras.Model(inputs = board_input, outputs=[move_output, score_output])

def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()
    else:
        model = load_model(model_name, custom_objects={'my_categorical_crossentropy': my_categorical_crossentropy})

    model.summary()

    # opt1 = tf.train.AdamOptimizer()
    opt1 = keras.optimizers.Adam()

    model.compile(optimizer=opt1,
                   loss={'moves': my_categorical_crossentropy, 'score': 'mean_squared_error' },
                   metrics=['accuracy', 'mae'])
    return model

repr = Repr2D()


pgn = open(sys.argv[1])
model_name = None
if len(sys.argv) > 2:
    model_name = sys.argv[2]

model = load_or_create_model(model_name)

train_data = np.zeros(((BATCH_SIZE, 8, 8, 17)), np.int8)
train_labels1 = np.zeros((BATCH_SIZE, 4672), np.int8)
train_labels2 = np.zeros((BATCH_SIZE, 1), np.float32)
cnt = 0

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
    white = game.headers["White"]
    black = game.headers["Black"]

    print("{}: {} - {}, {}". format(ngames, white, black, result))

    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))

    try:
        for move in game.main_line():
            train_data[cnt] = repr.board_to_array(b)
            train_labels1[cnt] = repr.move_to_array(b, move)
            train_labels2[cnt, 0] = label_for_result(result, b.turn)
            cnt += 1

            if cnt == BATCH_SIZE:
                train_labels = [ train_labels1, train_labels2 ]
                results = model.train_on_batch(train_data, train_labels)
                samples += cnt
                print(samples, results)
                cnt = 0
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

# Train on the remainder of the dataset
train_labels = [ train_labels1[:cnt], train_labels2[:cnt] ]
results = model.train_on_batch(train_data[:cnt], train_labels)
samples += cnt
print(samples, results)

if model_name is None:
    model.save("combined-model.h5")
else:
    model.save(model_name)
