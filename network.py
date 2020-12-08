# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

import math

WEIGHT_REGULARIZER = keras.regularizers.l2(1e-4)
ACTIVITY_REGULARIZER = None  # keras.regularizers.l1(1e-6)
RECTIFIER = "relu"
SIGMOID=tf.keras.activations.hard_sigmoid

INITIAL_LEARN_RATE = 1e-2
MIN_LEARN_RATE = 2e-5

SQUEEZE_EXCITE=False

def categorical_crossentropy_from_logits(target, output):
    return K.categorical_crossentropy(target, output, from_logits=True)


def residual_block(y, dim, index, residual=True):
    shortcut = y

    y = keras.layers.Activation(
        name="residual-block-{}-preactivation1".format(index), activation=RECTIFIER
    )(y)

    y = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="residual-block-{}-conv1".format(index),
        kernel_regularizer=WEIGHT_REGULARIZER,
        # activity_regularizer=ACTIVITY_REGULARIZER,
        activation="linear",
    )(y)

    y = keras.layers.Activation(
        name="residual-block-{}-preactivation2".format(index), activation=RECTIFIER
    )(y)

    y = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="residual-block-{}-conv2".format(index),
        kernel_regularizer=WEIGHT_REGULARIZER,
        # activity_regularizer=ACTIVITY_REGULARIZER,
        activation="linear",
    )(y)

    if SQUEEZE_EXCITE:
        t = keras.layers.GlobalAveragePooling2D(
            name="residual-block-{}-pooling".format(index)
        )(y)
        t = keras.layers.Dense(
            8,
            name="residual-block-{}-squeeze".format(index),
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=RECTIFIER,
        )(t)
        t = keras.layers.Dense(
            dim,
            name="residual-block-{}-excite".format(index),
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=SIGMOID,
        )(t)

        y = keras.layers.Multiply(name="residual-block-{}-multiply".format(index))([y, t])

    if residual:
        y = keras.layers.add([y, shortcut], name="residual-block-{}-add".format(index))

    return y


def create_policy_head(input):
    dim = input.shape.as_list()[-1]

    temp = keras.layers.Activation(
        name="pre-moves-activation", activation=RECTIFIER
    )(input)

    temp = keras.layers.Conv2D(
        dim,
        (3, 3),
        activation="linear",
        name="pre-moves-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        padding="same",
    )(temp)

    if SQUEEZE_EXCITE:
        t = keras.layers.GlobalAveragePooling2D(name="moves-pooling")(temp)
        t = keras.layers.Dense(
            8,
            name="moves-squeeze",
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=RECTIFIER,
        )(t)
        t = keras.layers.Dense(
            dim,
            name="moves-excite",
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=SIGMOID,
        )(t)

        temp = keras.layers.Multiply(name="moves-multiply")([temp, t])

    temp = keras.layers.add([temp, input], name="pre-moves-conv-add")

    temp = keras.layers.Conv2D(
        73,
        (3, 3),
        activation="linear",
        name="moves-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activity_regularizer=ACTIVITY_REGULARIZER,
        padding="same",
    )(temp)

    return keras.layers.Flatten(name="moves")(temp)


def create_value_head(input):
    dim = input.shape.as_list()[-1]

    if SQUEEZE_EXCITE:
        t = keras.layers.GlobalAveragePooling2D(name="value-pooling")(input)
        t = keras.layers.Dense(
            8,
            name="value-squeeze",
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=RECTIFIER,
        )(t)
        t = keras.layers.Dense(
            dim,
            name="value-excite",
            kernel_regularizer=WEIGHT_REGULARIZER,
            activation=SIGMOID,
        )(t)

        temp = keras.layers.Multiply(name="value-multiply")([input, t])
    else:
        temp = input

    temp = keras.layers.Conv2D(
        16,
        (1, 1),
        padding="same",
        name="pre-value-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation=RECTIFIER,
    )(temp)

    temp = keras.layers.Flatten(name="flatten-value")(temp)
    temp = keras.layers.BatchNormalization(name="value-dense-bn")(temp)
    temp = keras.layers.Dense(
        128,
        name="value-dense",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activity_regularizer=ACTIVITY_REGULARIZER,
    )(temp)

    temp = keras.layers.BatchNormalization(name="value-bn")(temp)
    eval_head = keras.layers.Dense(
        1, activation="tanh", kernel_regularizer=WEIGHT_REGULARIZER, name="value"
    )(temp)

    return eval_head


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape=(8, 8, repr.num_planes), name="board-input")

    layers = [[128, 9]]

    dim = layers[0][0]
    temp = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="initial-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        # activity_regularizer=ACTIVITY_REGULARIZER,
        activation="linear",
    )(board_input)

    index = 1
    residual = True

    for width, count in layers:
        for i in range(count):
            temp = keras.layers.BatchNormalization(
                name="residual-block-{}-bn".format(index)
            )(temp)
            temp = residual_block(temp, width, index, residual)
            index += 1
            residual = True
        residual = False

    temp = keras.layers.BatchNormalization(name="residual-block-{}-bn".format(index))(
        temp
    )

    move_output = create_policy_head(temp)
    value_output = create_value_head(temp)

    return keras.Model(
        name="TFlite{}_{}".format(
            "_SE" if SQUEEZE_EXCITE else "",
            "-".join(["{}x{}".format(width, count) for width, count in layers])
        ),
        inputs=[board_input],
        outputs=[move_output, value_output],
    )


def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()

        # optimizer = AdaBelief(lr=2e-3, weight_decay=1e-4)
        # optimizer = keras.optimizers.SGD(
        #     lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True, clipnorm=1.0
        # )
        optimizer = keras.optimizers.SGD(
            lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True
        )
        # optimizer = keras.optimizers.Adam(lr=0.001)

        model.compile(
            optimizer=optimizer,
            loss={
                "moves": categorical_crossentropy_from_logits,
                "value": "mean_squared_error",
            },
            loss_weights={"moves": 1.0, "value": 1.0, },
            metrics={
                "moves": ["accuracy", "top_k_categorical_accuracy"],
                "value": ["mae"],
            },
        )
    else:
        print('Loading model from "{}"'.format(model_name))
        model = load_model(
            model_name,
            custom_objects={
                "categorical_crossentropy_from_logits": categorical_crossentropy_from_logits,
            },
        )

    model.summary()
    print()
    print('Model name is "{}"'.format(model.name))
    print()

    return model


SAMPLE_RATE = 0.2
N_POSITIONS = 1330249
BATCH_SIZE = 256
STEPS_PER_ITERATION = SAMPLE_RATE * N_POSITIONS / BATCH_SIZE

def schedule_learn_rate(model, iteration, batch_no):

    t = iteration + batch_no / STEPS_PER_ITERATION
    learn_rate = MIN_LEARN_RATE + (INITIAL_LEARN_RATE - MIN_LEARN_RATE) * 0.5 * (
        1 + math.cos(t / 6 * math.pi)
    )

    K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
