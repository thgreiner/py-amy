import argparse

import tensorflow as tf
import network
import numpy as np
import pickle
from random import randint

from network import load_or_create_model

import logging

SAMPLE = 5


def representative_dataset_gen():
    with open("data/validation.pkl", "rb") as fin:
        try:
            cnt = 0
            while cnt < 1000:
                item = pickle.load(fin)
                if randint(0, 99) < SAMPLE:
                    features = item.data_board.reshape(1, 8, 8, 19).astype("float32")
                    yield [features]
                    cnt += 1
        except EOFError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize a Keras model.")
    parser.add_argument("--model", help="model file name")

    args = parser.parse_args()

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
