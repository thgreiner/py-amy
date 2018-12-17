import sys
import random

import chess
from chess import Board, Move, Piece
import chess.pgn

from chess_input import Repr2D

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 1_100_000
OPENING = 8
MODEL_NAME='model-resnet.h5'


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


def phasing(label, moves_in_game, current_move):
    # return label * (1.0 + moves_in_game - current_move) ** -0.8
    return label * (0.97 ** (moves_in_game - current_move))


def add_common_layers(y):
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)
    return y

cardinality = 4

def grouped_convolution(y, nb_channels = 96, _strides = (1,1)):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return keras.layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = keras.layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(keras.layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = keras.layers.concatenate(groups)

    return y

def residual_block(y, nb_channels_in=96, nb_channels_out=96, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = keras.layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    y = keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = keras.layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    y = keras.layers.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = keras.layers.LeakyReLU()(y)

    return y


def create_model():
    board_input = keras.layers.Input(shape = (8, 8, 12), name='board_input')
    castle_input =  keras.layers.Input(shape = (4,), name='castle_input')

    conv1 = keras.layers.Conv2D(96, (3, 3), name='conv1', padding='same')(board_input)
    tmp = add_common_layers(conv1)

    tmp = residual_block(tmp)
    tmp = residual_block(tmp)
    tmp = residual_block(tmp)
    tmp = residual_block(tmp)

    tmp = keras.layers.Conv2D(4, (1, 1), name='after_res', padding='same')(tmp)
    tmp = add_common_layers(tmp)
    
    flatten = keras.layers.Flatten(name='flatten')(tmp)
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

    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'])

    train_data, train_labels = parse_pgn_to_training_data(file_name)

    callback = keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    while True:
        history = model.fit(train_data, train_labels, batch_size=128, epochs=100,
                             validation_split = 0.1,
                             callbacks=[callback])
        model.save(MODEL_NAME)



if __name__ == '__main__':
    train_model_from_pgn(sys.argv[1])
