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

repr = Repr2D()
LEAK=0.1

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 750_000


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0

def create_model():
    board_input = keras.layers.Input(shape = (8, 8, 12), name='board_input')

    dim = 64

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same')(board_input)
    temp = keras.layers.BatchNormalization()(temp)
    temp = keras.layers.LeakyReLU()(temp)

    for i in range(8):
        temp = keras.layers.Conv2D(dim, (3, 3), padding='same')(temp)
        temp = keras.layers.BatchNormalization()(temp)
        temp = keras.layers.LeakyReLU()(temp)

    output = keras.layers.Conv2D(7, (3, 3), activation='sigmoid', padding='same', name='output')(temp)

    return keras.Model(inputs = board_input, outputs=output)


model = create_model()
model.summary()

# opt1 = tf.train.AdamOptimizer()
opt1 = keras.optimizers.Adam()

model.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['accuracy'])

pgn = open("PGN/TWIC.pgn")
# pgn = open("LearnGames.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data = np.zeros(((npos, 8, 8, 12)), np.int8)
train_labels = np.zeros((npos, 8, 8, 7), np.int8)
cnt1 = 0

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
            train_data[cnt1] = repr.board_to_array(b)
            train_labels[cnt1] = repr.move_to_array(b, piece, move)
            cnt1 += 1

        b.push(move)
        if cnt1 >= npos:
            break
    if cnt1 >= npos:
        break
    print("{}: {}".format(ngames, cnt1), end='\r')

print("{} {}".format(ngames, cnt1))

if cnt1 < npos:
    train_data = train_data[:cnt1]
    train_labels = train_labels[:cnt1]

save_callback = keras.callbacks.ModelCheckpoint("weights-move-{epoch:02d}-{val_loss:.2f}.hdf5")

while True:
    history = model.fit(train_data, train_labels, batch_size=256, epochs=25,
        validation_split=0.1,
        callbacks=[save_callback])
    model.save("move-model.h5")
