# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

# We really need almost no regularization as the model has so few params
REGULARIZATION_WEIGHT=1e-4

L2_REGULARIZER = None # keras.regularizers.l2(REGULARIZATION_WEIGHT)

RECTIFIER='relu'

def residual_block(y, dim, index, factor=3):
    shortcut = y

    y = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(y)
    y = keras.layers.Conv2D(factor * dim, (1, 1),
                            padding='same',
                            name="residual-block-{}-expand".format(index),
                            kernel_initializer='lecun_normal',
                            activation=RECTIFIER)(y)

    y = keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                             name="residual-block-{}-depthwise".format(index),
                                             kernel_initializer='lecun_normal',
                                             activation=RECTIFIER)(y)

    y = keras.layers.Conv2D(dim, (1, 1), padding='same',
                                         name="residual-block-{}-contract".format(index),
                                         kernel_initializer='lecun_normal',
                                         activation='linear')(y)

    y = keras.layers.add([y, shortcut], name="residual-block-{}-add".format(index))
    return y


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape = (8, 8, repr.num_planes), name='board-input')
    moves_input = keras.layers.Input(shape = (4672,), name='moves-input')
    non_progress_input = keras.layers.Input(shape = (1,), name='non-progress-input')

    dim = 112

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same',
                                            name="initial-conv",
                                            kernel_regularizer=L2_REGULARIZER,
                                            bias_regularizer=L2_REGULARIZER,
                                            kernel_initializer='lecun_normal',
                                            activation=RECTIFIER)(board_input)

    index = 1

    for i in range(13):
        temp = residual_block(temp, dim, index)
        index += 1


    # Create the policy head
    t2 = residual_block(temp, dim, index)

    t2 = keras.layers.BatchNormalization(name="pre-moves-bn")(t2)
    t2 = keras.layers.Conv2D(73, (3, 3), activation='linear',
                                         name="pre-moves-conv",
                                         padding='same')(t2)

    t2 = keras.layers.Flatten(name='flatten-moves')(t2)
    t2 = keras.layers.multiply([t2, moves_input], name='limit-to-legal-moves')
    move_output = keras.layers.Activation("softmax", name='moves')(t2)

    # Create the value head
    temp = keras.layers.BatchNormalization(name="pre-value-bn".format(index))(temp)
    temp = keras.layers.Conv2D(9, (1, 1), padding='same',
                                          name="pre-value-conv",
                                          kernel_regularizer=L2_REGULARIZER,
                                          bias_regularizer=L2_REGULARIZER,
                                          kernel_initializer='lecun_normal',
                                          activation=RECTIFIER)(temp)
    temp = keras.layers.Flatten(name="flatten-value")(temp)
    temp = keras.layers.concatenate([temp, non_progress_input], name="concat-non-progress")
    temp = keras.layers.BatchNormalization(name="value-dense-bn")(temp)
    temp = keras.layers.Dense(128,
                              name="value-dense",
                              kernel_regularizer=L2_REGULARIZER,
                              bias_regularizer=L2_REGULARIZER,
                              kernel_initializer='lecun_normal',
                              activation=RECTIFIER)(temp)

    temp = keras.layers.BatchNormalization(name="value-bn")(temp)
    value_output = keras.layers.Dense(1, activation='tanh',
                                         kernel_regularizer=L2_REGULARIZER,
                                         bias_regularizer=L2_REGULARIZER,
                                         name='value')(temp)

    return keras.Model(
        name = "MobileNet V2-like (BN, RELU)",
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
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
                  loss={'moves': 'categorical_crossentropy', 'value': 'mean_squared_error' },
                  metrics=['accuracy', 'mae'])
    return model


def schedule_learn_rate(model, batch_no):

    initial_learn_rate = 0.02
    
    learn_rate = initial_learn_rate * 0.9 ** (batch_no / 1000)
    
    K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
