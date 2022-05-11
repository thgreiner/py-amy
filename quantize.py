import argparse

import tensorflow as tf
import network
import numpy as np
import pickle
from random import randint

from network import load_or_create_model

import logging

from chess import Board
from chess_input import Repr2D

import tensorflow_model_optimization as tfmot
from glob_p import find_train_files

SAMPLE = 5

repr = Repr2D()


def representative_dataset_gen():
    yield [repr.board_to_array(Board()).reshape(1, 8, 8, 19).astype("float32")]

    files = find_train_files(600_000, 10, True)
    cnt = 0

    for filename in files:
        with open(filename, "rb") as fin:
            while cnt < 200:
                try:
                    item = pickle.load(fin)
                    if randint(0, 99) < SAMPLE:
                        features = item.data_board.reshape(1, 8, 8, 19).astype(
                            "float32"
                        )
                        yield [features]
                        cnt += 1
                        print(cnt, end="\r")
                except EOFError:
                    pass

    print(f"Provided {cnt} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize a Keras model.")
    parser.add_argument("--model", help="model file name")

    args = parser.parse_args()

    with tfmot.quantization.keras.quantize_scope():
        model = load_or_create_model(args.model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(args.model.replace("h5", "tflite"), "wb") as fout:
        fout.write(tflite_quant_model)
