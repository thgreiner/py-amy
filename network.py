# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from chess_input import Repr2D

import math

WEIGHT_REGULARIZER = keras.regularizers.l2(1e-4)
RECTIFIER = "relu"
RENORM = True

INITIAL_LEARN_RATE = 1e-2
MIN_LEARN_RATE = 1e-4

WDL_WEIGHT = 0.1
MLH_WEIGHT = 0.025

def residual_block(input, dim, index, residual=True):

    flow = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="residual-block-{}-conv1".format(index),
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation="linear",
    )(input)

    flow = keras.layers.BatchNormalization(
        name="residual-block-{}-bn1".format(index),
        renorm=RENORM,
    )(flow)

    flow = keras.layers.Activation(
        name="residual-block-{}-activation1".format(index), activation=RECTIFIER
    )(flow)

    flow = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="residual-block-{}-conv2".format(index),
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation="linear",
    )(flow)

    flow = keras.layers.BatchNormalization(
        name="residual-block-{}-bn2".format(index),
        renorm=RENORM,
    )(flow)

    flow = keras.layers.Activation(
        name="residual-block-{}-activation2".format(index), activation=RECTIFIER
    )(flow)

    if residual:
        flow = keras.layers.add(
            [flow, input], name="residual-block-{}-add".format(index)
        )

    return flow


def create_policy_head(input):
    dim = input.shape.as_list()[-1]

    flow = keras.layers.Conv2D(
        dim,
        (3, 3),
        activation="linear",
        name="pre-moves-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        padding="same",
    )(input)

    flow = keras.layers.Activation(name="pre-moves-activation", activation=RECTIFIER)(
        flow
    )

    flow = keras.layers.add([flow, input], name="pre-moves-conv-add")

    flow = keras.layers.Conv2D(
        73,
        (3, 3),
        activation="linear",
        name="moves-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        padding="same",
    )(flow)

    return keras.layers.Flatten(name="moves")(flow)


def create_value_head(input):
    flow = keras.layers.Conv2D(
        32,
        (1, 1),
        padding="same",
        name="pre-value-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation=RECTIFIER,
    )(input)

    flow = keras.layers.Flatten(name="flatten-value")(flow)
    flow = keras.layers.BatchNormalization(name="value-dense-bn", renorm=RENORM)(flow)

    flow = keras.layers.Dense(
        128,
        name="value-dense",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation=RECTIFIER,
    )(flow)

    flow = keras.layers.BatchNormalization(name="value-bn", renorm=RENORM)(flow)

    eval_head = keras.layers.Dense(
        1, activation="tanh", kernel_regularizer=WEIGHT_REGULARIZER, name="value"
    )(flow)

    wdl_head = keras.layers.Dense(
        3, activation="softmax", kernel_regularizer=WEIGHT_REGULARIZER, name="wdl"
    )(flow)

    return eval_head, wdl_head


def create_moves_left_head(input):
    flow = keras.layers.Conv2D(
        8,
        (1, 1),
        padding="same",
        name="pre-mlh-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation=RECTIFIER,
    )(input)

    flow = keras.layers.Flatten(name="flatten-mlh")(flow)
    flow = keras.layers.BatchNormalization(name="mlh-dense-bn", renorm=RENORM)(flow)

    flow = keras.layers.Dense(
        64,
        name="mlh-dense",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation=RECTIFIER,
    )(flow)

    flow = keras.layers.BatchNormalization(name="mlh-bn", renorm=RENORM)(flow)

    mlh_head = keras.layers.Dense(
        1, activation=RECTIFIER, kernel_regularizer=WEIGHT_REGULARIZER, name="mlh"
    )(flow)

    return mlh_head


def create_model():
    repr = Repr2D()

    board_input = keras.layers.Input(shape=(8, 8, repr.num_planes), name="board-input")

    layers = [[128, 10]]

    dim = layers[0][0]
    flow = keras.layers.Conv2D(
        dim,
        (3, 3),
        padding="same",
        name="initial-conv",
        kernel_regularizer=WEIGHT_REGULARIZER,
        activation="linear",
    )(board_input)

    index = 1
    residual = True

    for width, count in layers:
        for i in range(count):
            flow = residual_block(flow, width, index, residual)
            index += 1
            residual = True
        residual = False

    flow = keras.layers.BatchNormalization(
        name="residual-block-{}-bn".format(index), renorm=RENORM
    )(flow)

    move_output = create_policy_head(flow)
    value_output, wdl_output = create_value_head(flow)
    mlh_output = create_moves_left_head(flow)

    return keras.Model(
        name="TFlite_{}".format(
            "-".join(["{}x{}".format(width, count) for width, count in layers])
        ),
        inputs=[board_input],
        outputs=[move_output, value_output, wdl_output, mlh_output],
    )


def compile_model(model, prefix=""):
    optimizer = keras.optimizers.SGD(
        lr=INITIAL_LEARN_RATE, momentum=0.9, nesterov=True, clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            f"{prefix}moves": keras.losses.CategoricalCrossentropy(from_logits=True),
            "value": "mean_squared_error",
            f"{prefix}wdl": keras.losses.CategoricalCrossentropy(),
            f"{prefix}mlh": keras.losses.MeanSquaredLogarithmicError(),
        },
        metrics={
            f"{prefix}moves": ["accuracy", "top_k_categorical_accuracy"],
            "value": ["mae"],
            f"{prefix}wdl": ["accuracy"],
            f"{prefix}mlh": ["mae"],
        },
        loss_weights={
            f"{prefix}moves": 1.0,
            "value": 1.0,
            f"{prefix}wdl": WDL_WEIGHT,
            f"{prefix}mlh": MLH_WEIGHT,
        },
    )


def load_or_create_model(model_name):
    if model_name is None:
        model = create_model()
        compile_model(model)
    else:
        print('Loading model from "{}"'.format(model_name))
        model = load_model(
            model_name,
        )

    # model.summary()
    # print()
    # print('Model name is "{}"'.format(model.name))
    # print()

    return model


SAMPLE_RATE = 0.2
N_POSITIONS = 1_000_000
BATCH_SIZE = 256
STEPS_PER_ITERATION = SAMPLE_RATE * N_POSITIONS / BATCH_SIZE


def schedule_learn_rate(model, iteration, batch_no):

    # t = iteration + batch_no / STEPS_PER_ITERATION
    # learn_rate = MIN_LEARN_RATE + (INITIAL_LEARN_RATE - MIN_LEARN_RATE) * 0.5 * (
    #     1 + math.cos(t / 6 * math.pi)
    # )

    learn_rate = 2e-3
    K.set_value(model.optimizer.lr, learn_rate)
    return learn_rate
