from queue import Queue
from network import load_or_create_model
import numpy as np
from threading import Event

class DeferredEvaluator:

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.features_board = np.zeros(((self.batch_size, 8, 8, 18)), np.int8)
        self.features_non_progress = np.zeros((self.batch_size, 1), np.float32)
        self.i = 0

    def add(self, features):
        self.features_board[self.i] = features[0]
        self.features_non_progress[self.i] = features[1]
        self.i += 1

    def evaluate(self):
        # print("evaluate {}".format(self.i))
        predictions = self.model.predict([self.features_board[:self.i], self.features_non_progress[:self.i]])
        for i in range(self.i):
            yield((predictions[0][i], predictions[1][i]))

    def clear(self):
        self.i = 0
