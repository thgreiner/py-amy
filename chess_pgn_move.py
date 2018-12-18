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
POSITIONS_TO_LEARN_APRIORI = 250_000


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


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
        group = keras.layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d], output_shape=(8, 8, 24))(y)
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

    conv1 = keras.layers.Conv2D(96, (3, 3), name='conv1', padding='same')(board_input)
    temp = add_common_layers(conv1)

    temp = residual_block(temp)
    temp = residual_block(temp)
    temp = residual_block(temp)
    temp = residual_block(temp)

    output = keras.layers.Conv2D(7, (3, 3), activation='relu', padding='same', name='output')(temp)

    return keras.Model(inputs = board_input, outputs=output)


model = create_model()
model.summary()

# opt1 = tf.train.AdamOptimizer()
opt1 = keras.optimizers.Adam()

model.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

pgn = open("PGN/ClassicGames.pgn")
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
