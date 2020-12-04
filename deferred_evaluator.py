import numpy as np
from prometheus_client import Histogram


class DeferredEvaluator:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.features_board = np.zeros(((self.batch_size, 8, 8, 19)), np.int8)
        self.i = 0
        self.parallelism_histogram = Histogram(
            "parallelism", "Parallelism", buckets=[i for i in range(0, batch_size + 1)]
        )

    def add(self, features):
        self.features_board[self.i] = features
        self.i += 1

    def evaluate(self):
        self.parallelism_histogram.observe(self.i)
        # print("evaluate {}".format(self.i))
        predictions = self.model.predict(self.features_board[: self.i])
        for i in range(self.i):
            yield ((predictions[0][i], predictions[1][i]))

    def clear(self):
        self.i = 0
