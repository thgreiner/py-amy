# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

# We really need almost no regularization as the model has so few params
REGULARIZATION_WEIGHT=2e-5

L2_REGULARIZER = keras.regularizers.l2(REGULARIZATION_WEIGHT)

RECTIFIER='elu'

INITIAL_LEARN_RATE = 0.01

def categorical_crossentropy_from_logits(target, output):
    return K.categorical_crossentropy(target, output, from_logits=True)


def residual_block(y, dim, index, residual=True, factor=3):
    shortcut = y

    # y = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(y)

    y = keras.layers.Conv2D(factor * dim, (1, 1),
                            padding='same',
                            name="residual-block-{}-expand".format(index),
                            kernel_regularizer=L2_REGULARIZER,
                            activation=RECTIFIER)(y)

    y = keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                             name="residual-block-{}-depthwise".format(index),
                                             kernel_regularizer=L2_REGULARIZER,
                                             activation=RECTIFIER)(y)

    y = keras.layers.Conv2D(dim, (1, 1), padding='same',
                                         name="residual-block-{}-contract".format(index),
                                         kernel_regularizer=L2_REGULARIZER,
                                         activation='linear')(y)

    if residual:
        y = keras.layers.add([y, shortcut], name="residual-block-{}-add".format(index))

    return y

def create_policy_head(input):
    dim = input.shape.as_list()[-1]

    temp = keras.layers.Conv2D(dim, (3, 3), activation='linear',
                                            name="pre-moves-conv",
                                            kernel_regularizer=L2_REGULARIZER,
                                            padding='same')(input)
    temp = keras.layers.add([temp, input], name="pre-moves-conv-add")
    temp = keras.layers.Activation(name='pre-moves-activation', activation=RECTIFIER)(temp)

    temp = keras.layers.Conv2D(73, (3, 3), activation='linear',
                                           name="moves-conv",
                                           kernel_regularizer=L2_REGULARIZER,
                                           padding='same')(temp)

    return keras.layers.Flatten(name='moves')(temp)

def create_value_head(input, non_progress_input):
    temp = keras.layers.Conv2D(9, (1, 1), padding='same',
                                          name="pre-value-conv",
                                          kernel_regularizer=L2_REGULARIZER,
                                          activation=RECTIFIER)(input)
    temp = keras.layers.Flatten(name="flatten-value")(temp)
    temp = keras.layers.concatenate([temp, non_progress_input], name="concat-non-progress")
    temp = keras.layers.BatchNormalization(name="value-dense-bn")(temp)
    temp = keras.layers.Dense(128,
                              name="value-dense",
                              kernel_regularizer=L2_REGULARIZER,
                              activation=RECTIFIER)(temp)

    temp = keras.layers.BatchNormalization(name="value-bn")(temp)
    return keras.layers.Dense(1, activation='tanh',
                                 kernel_regularizer=L2_REGULARIZER,
                                 name='value')(temp)

def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape = (8, 8, repr.num_planes), name='board-input')
    non_progress_input = keras.layers.Input(shape = (1,), name='non-progress-input')

    dim = 64

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same',
                                            name="initial-conv",
                                            kernel_regularizer=L2_REGULARIZER,
                                            activation=RECTIFIER)(board_input)


    index = 1
    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index)
        index += 1

    dim = 96
    residual = False

    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index, residual)
        index += 1
        residual = True

    dim = 128
    residual = False

    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index, residual)
        index += 1
        residual = True


    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)

    move_output = create_policy_head(temp)
    value_output = create_value_head(temp, non_progress_input)

    return keras.Model(
        name = "MobileNet V2-like (BN, ELU, Improved Scale-Up layer)",
        inputs = [board_input, non_progress_input],
        outputs = [move_output, value_output])


def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()
    else:
        print("Loading model from \"{}\"".format(model_name))
        model = load_model(model_name, custom_objects={'categorical_crossentropy_from_logits': categorical_crossentropy_from_logits})

    model.summary()
    print()
    print("Model name is \"{}\"".format(model.name))
    print()

    # optimizer = keras.optimizers.Adam(lr = 0.002)
    optimizer = keras.optimizers.SGD(lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
                  loss={'moves': categorical_crossentropy_from_logits, 'value': 'mean_squared_error' },
                  metrics=['accuracy', 'mae'])
    return model


def schedule_learn_rate(model, batch_no):

    learn_rate = INITIAL_LEARN_RATE * 0.97 ** (batch_no / 1000)

    K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
