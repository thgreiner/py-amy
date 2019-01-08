# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

# We really need almost no regularization as the model has so few params
REGULARIZATION_WEIGHT=1e-6

def residual_block(y, dim):
    shortcut = y
    y = keras.layers.Conv2D(dim, (1, 1), padding='same',
                                         activation='elu',
                                         kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(y)
    y = keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                             activation='elu',
                                             kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(y)
    y = keras.layers.Conv2D(dim, (1, 1), padding='same',
                                         activation='elu',
                                         kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(y)
    y = keras.layers.add([y, shortcut])
    return y


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape = (8, 8, repr.num_planes), name='board_input')
    moves_input = keras.layers.Input(shape = (4672,), name='moves_input')

    dim = 128

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same',
                                            activation='elu',
                                            kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(board_input)

    for i in range(15):
        temp = residual_block(temp, dim)


    t2 = keras.layers.Conv2D(128, (3, 3), padding='same',
                                          activation='elu',
                                          kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(temp)

    t2 = keras.layers.Conv2D(73, (3, 3), activation='linear',
                                         padding='same',
                                         kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(t2)
    t2 = keras.layers.Flatten()(t2)
    t2 = keras.layers.multiply([t2, moves_input])
    move_output = keras.layers.Activation("softmax", name='moves')(t2)

    temp = keras.layers.Conv2D(8, (1, 1), padding='same',
                                          activation='elu',
                                          kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(temp)
    temp = keras.layers.Flatten()(temp)
    temp = keras.layers.Dense(192, activation='elu',
                                   kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT))(temp)

    score_output = keras.layers.Dense(1, activation='tanh',
                                         kernel_regularizer=keras.regularizers.l2(REGULARIZATION_WEIGHT),
                                         name='score')(temp)

    return keras.Model(
        name = "ResNet-like",
        inputs = [board_input, moves_input],
        outputs=[move_output, score_output])


def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()
    else:
        print("Loading model from \"{}\"".format(model_name))
        model = load_model(model_name)

    model.summary()
    print()
    print("Model name is \"{}\"".format(model.name))
    print()

    optimizer = keras.optimizers.Adam()
    # optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss={'moves': 'categorical_crossentropy', 'score': 'mean_squared_error' },
                  metrics=['accuracy', 'mae'])
    return model
