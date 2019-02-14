# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

# We really need almost no regularization as the model has so few params
REGULARIZATION_WEIGHT=1e-4

L2_REGULARIZER = None # keras.regularizers.l2(REGULARIZATION_WEIGHT)

RECTIFIER='elu'

BN=False

def residual_block(y, dim, index):
    shortcut = y

    dim2 = dim // 2

    if BN:
        y = keras.layers.BatchNormalization()(y)

    cardinality = 4
    _d = dim2 // cardinality

    y = keras.layers.Conv2D(dim2, (1, 1), padding='same',
                                          name="residual-block-{}-bottleneck".format(index),
                                          use_bias=False,
                                          activation=RECTIFIER)(y)

    groups = []
    for j in range(cardinality):
        group = keras.layers.Lambda(
            lambda z: z[:, :, :, j * _d:j * _d + _d],
            name="residual-block-{}-group-{}".format(index, j))(y)
        groups.append(keras.layers.Conv2D(_d, kernel_size=(3, 3),
            activation=RECTIFIER, padding='same',
            name="residual-block-{}-group-{}-conv2d".format(index, j))(group))

    y = keras.layers.concatenate(groups, name="residual-block-{}-concat".format(index))

    y = keras.layers.Conv2D(dim, (1, 1), padding='same',
                                         name="residual-block-{}-mix".format(index),
                                         use_bias=False,
                                         activation='linear')(y)

    y = keras.layers.add([y, shortcut], name="residual-block-{}-add".format(index))
    y = keras.layers.Activation(RECTIFIER, name="residual-block-{}-activation".format(index))(y)

    return y


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape = (8, 8, repr.num_planes), name='board-input')
    moves_input = keras.layers.Input(shape = (4672,), name='moves-input')
    non_progress_input = keras.layers.Input(shape = (1,), name='non-progress-input')

    dim = 160

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same',
                                            name="initial-conv",
                                            activation=RECTIFIER)(board_input)

    index = 1
    for i in range(17):
        temp = residual_block(temp, dim, index)
        index += 1



    # Create the policy head
    if BN:
        t2 = keras.layers.BatchNormalization()(temp)
    else:
        t2 = temp

    t2 = keras.layers.Conv2D(dim, (3, 3), name="policy-head-conv",
                                          activation=RECTIFIER,
                                          padding='same')(t2)
    t2 = keras.layers.Conv2D(73, (3, 3), activation='linear',
                                         name="pre-moves-conv",
                                         padding='same')(t2)

    t2 = keras.layers.Flatten(name='flatten-moves')(t2)
    t2 = keras.layers.multiply([t2, moves_input], name='limit-to-legal-moves')
    move_output = keras.layers.Activation("softmax", name='moves')(t2)

    # Create the value head
    if BN:
        temp = keras.layers.BatchNormalization()(temp)

    temp = keras.layers.Conv2D(9, (1, 1), padding='same',
                                          name="pre-value-conv",
                                          activation=RECTIFIER)(temp)
    temp = keras.layers.Flatten(name="flatten-value")(temp)
    temp = keras.layers.concatenate([temp, non_progress_input], name="concat-non-progress")
    if BN:
        temp = keras.layers.BatchNormalization()(temp)

    temp = keras.layers.Dense(128,
                              name="value-dense",
                              activation=RECTIFIER)(temp)

    if BN:
        temp = keras.layers.BatchNormalization()(temp)

    value_output = keras.layers.Dense(1, activation='tanh',
                                         kernel_initializer='random_uniform',
                                         name='value')(temp)

    if BN:
        bn_prefix = ""
    else:
        bn_prefix = "No "

    return keras.Model(
        name = "Grouped Convs ({}BN, {})".format(bn_prefix, RECTIFIER),
        inputs = [board_input, moves_input, non_progress_input],
        outputs = [move_output, value_output])


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

    # optimizer = keras.optimizers.Adam(lr = 0.001)
    optimizer = keras.optimizers.SGD(lr=0.002, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
                  loss={'moves': 'categorical_crossentropy', 'value': 'mean_squared_error' },
                  metrics=['accuracy', 'mae'])
    return model


def schedule_learn_rate(model, batch_no):

    min_rate = 0.02
    max_rate = 0.2

    peak = 1200

    if batch_no < peak // 2:
        learn_rate = min_rate + (batch_no / (peak // 2)) * (max_rate - min_rate)
    elif batch_no < peak:
        learn_rate = min_rate + ((peak - batch_no) / (peak // 2)) * (max_rate - min_rate)
    else:
        learn_rate = min_rate / (1 + (batch_no - 900) / 500)

    # K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
