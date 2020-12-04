import tensorflow as tf
import network
import numpy as np
import pickle
from random import randint

from network import load_or_create_model

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

def show_value(model, data):
    result = model.predict([data])
    print(result)

sample=5

model = load_or_create_model('tflite_80x9.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
    with open("data/validation.pkl", "rb") as fin:
        try:
            cnt=0
            while(cnt < 1000):
                item = pickle.load(fin)
                if randint(0, 99) < sample:
                    features = item.data_board.reshape(1,8,8,19).astype('float32')
                    print(features)
                    # show_value(model, features)
                    yield [features]
                    cnt += 1
        except EOFError:
            pass

converter.representative_dataset = representative_dataset_gen
# converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

tflite_quant_model = converter.convert()

with open("quantized-model", "wb") as fout:
    fout.write(tflite_quant_model)
