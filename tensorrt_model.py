import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np

class TensorRTModel:

    def __init__(self, input_saved_model="tensorrt-model"):

        saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        print(signature_keys)

        self.infer = saved_model_loaded.signatures['serving_default']
        print(self.infer.structured_outputs)

        self.name = "TensorRT"

    def predict(self, input_board):

        t = tf.constant(input_board.astype(np.float32))
        result = self.infer(t)

        return (result['moves'].numpy(), result['value'].numpy())


# Test

if __name__ == "__main__":

    m = TensorRTModel()

    from chess import Board
    from chess_input import Repr2D

    r = Repr2D()
    i = r.board_to_array(Board())

    i = np.expand_dims(i, axis=0)

    print(m.predict(i))
