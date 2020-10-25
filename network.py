# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

WEIGHT_REGULARIZER = keras.regularizers.l2(1e-4)
ACTIVITY_REGULARIZER = None # keras.regularizers.l1(1e-6)
RECTIFIER='elu'

INITIAL_LEARN_RATE = 1e-2

def categorical_crossentropy_from_logits(target, output):
    return K.categorical_crossentropy(target, output, from_logits=True)


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def residual_block(y, dim, index, residual=True, factor=4):
    shortcut = y

    y = keras.layers.Conv2D(dim, (3, 3),
                            padding='same',
                            name="residual-block-{}-conv".format(index),
                            kernel_regularizer=WEIGHT_REGULARIZER,
                            # activity_regularizer=ACTIVITY_REGULARIZER,
                            activation=RECTIFIER)(y)

    t = keras.layers.GlobalAveragePooling2D(name="residual-block-{}-pooling".format(index))(y)
    t = keras.layers.Dense(8, name="residual-block-{}-squeeze".format(index),
                              kernel_regularizer=WEIGHT_REGULARIZER,
                              activation=RECTIFIER)(t)
    t = keras.layers.Dense(dim, name="residual-block-{}-excite".format(index),
                                kernel_regularizer=WEIGHT_REGULARIZER,
                                activation='sigmoid')(t)

    y = keras.layers.Multiply(name="residual-block-{}-multiply".format(index))([y, t])

    if residual:
        y = keras.layers.add([y, shortcut], name="residual-block-{}-add".format(index))

    return y

def create_policy_head(input):
    dim = input.shape.as_list()[-1]

    temp = keras.layers.Conv2D(dim, (3, 3), activation='linear',
                                            name="pre-moves-conv",
                                            kernel_regularizer=WEIGHT_REGULARIZER,
                                            padding='same')(input)

    t = keras.layers.GlobalAveragePooling2D(name="moves-pooling")(temp)
    t = keras.layers.Dense(8, name="moves-squeeze",
                              kernel_regularizer=WEIGHT_REGULARIZER,
                              activation=RECTIFIER)(t)
    t = keras.layers.Dense(dim, name="moves-excite",
                                kernel_regularizer=WEIGHT_REGULARIZER,
                                activation='sigmoid')(t)

    temp = keras.layers.Multiply(name="moves-multiply")([temp, t])

    temp = keras.layers.add([temp, input], name="pre-moves-conv-add")
    temp = keras.layers.Activation(name='pre-moves-activation', activation=RECTIFIER)(temp)

    temp = keras.layers.Conv2D(73, (3, 3), activation='linear',
                                           name="moves-conv",
                                           kernel_regularizer=WEIGHT_REGULARIZER,
                                           activity_regularizer=ACTIVITY_REGULARIZER,
                                           padding='same')(temp)

    return keras.layers.Flatten(name='moves')(temp)


def create_value_head(input, non_progress_input):
    dim = input.shape.as_list()[-1]

    t = keras.layers.GlobalAveragePooling2D(name="value-pooling")(input)
    t = keras.layers.Dense(8, name="value-squeeze",
                              kernel_regularizer=WEIGHT_REGULARIZER,
                              activation=RECTIFIER)(t)
    t = keras.layers.Dense(dim, name="value-excite",
                                kernel_regularizer=WEIGHT_REGULARIZER,
                                activation='sigmoid')(t)

    temp = keras.layers.Multiply(name="value-multiply")([input, t])

    temp = keras.layers.Conv2D(24, (1, 1), padding='same',
                                           name="pre-value-conv",
                                           kernel_regularizer=WEIGHT_REGULARIZER,
                                           activation=RECTIFIER)(temp)
    temp = keras.layers.Flatten(name="flatten-value")(temp)
    temp = keras.layers.concatenate([temp, non_progress_input], name="concat-non-progress")
    temp = keras.layers.BatchNormalization(name="value-dense-bn")(temp)
    temp = keras.layers.Dense(128,
                              name="value-dense",
                              kernel_regularizer=WEIGHT_REGULARIZER,
                              activity_regularizer=ACTIVITY_REGULARIZER,
                              activation=RECTIFIER)(temp)

    temp = keras.layers.BatchNormalization(name="value-bn")(temp)
    eval_head = keras.layers.Dense(1, activation='tanh',
                                   kernel_regularizer=WEIGHT_REGULARIZER,
                                   name='value')(temp)

    result_head = keras.layers.Dense(3, activation='softmax',
                                     kernel_regularizer=WEIGHT_REGULARIZER,
                                     name='result')(temp)
    return eval_head, result_head


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape = (8, 8, repr.num_planes), name='board-input')
    non_progress_input = keras.layers.Input(shape = (1,), name='non-progress-input')

    dim = 48

    temp = keras.layers.Conv2D(dim, (3, 3), padding='same',
                                            name="initial-conv",
                                            kernel_regularizer=WEIGHT_REGULARIZER,
                                            # activity_regularizer=ACTIVITY_REGULARIZER,
                                            activation=RECTIFIER)(board_input)


    index = 1
    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index, factor=6)
        index += 1

    dim = 64
    residual = False

    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index, residual)
        index += 1
        residual = True

    dim = 96
    residual = False

    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)
    for i in range(6):
        temp = residual_block(temp, dim, index, residual)
        index += 1
        residual = True

    temp  = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(temp)

    move_output = create_policy_head(temp)
    value_output, game_result_output = create_value_head(temp, non_progress_input)

    return keras.Model(
        name = "Full_Convolution_with_SE_48_64_96",
        inputs = [board_input, non_progress_input],
        outputs = [move_output, value_output, game_result_output])


def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()
    else:
        print("Loading model from \"{}\"".format(model_name))
        model = load_model(model_name, custom_objects={
            'categorical_crossentropy_from_logits': categorical_crossentropy_from_logits,
            'huber_loss': huber_loss })

    model.summary()
    print()
    print("Model name is \"{}\"".format(model.name))
    print()

    # optimizer = keras.optimizers.SGD(lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True, clipnorm=1.0)
    optimizer = keras.optimizers.SGD(lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.Adam(lr=0.001)
    # optimizer = AdaBound()

    model.compile(optimizer=optimizer,
                  loss={ 'moves': categorical_crossentropy_from_logits,
                         'value': "mean_squared_error",
                         'result': "categorical_crossentropy" },
                  loss_weights={ 'moves': 1.0, 'value': 1.0, 'result': 0.15},
                  metrics={ 'moves' :['accuracy'], 'value': ['mae'], 'result': ['accuracy'] })
    return model


def schedule_learn_rate(model, iteration, batch_no):

    learn_rate = INITIAL_LEARN_RATE # / (iteration + 1)

    K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
