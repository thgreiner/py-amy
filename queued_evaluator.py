from queue import Queue
from network import load_or_create_model
import numpy as np

class QueuedEvaluator:

    def __init__(self, target_queue: Queue):
        self.result_queue = Queue()
        self.target_queue = target_queue
        self.name = "bla"

    def predict(self, features):
        self.target_queue.put((self.result_queue, features))
        prediction = self.result_queue.get()
        return prediction


class MultiplexingEvaluator:

    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size

    def run(self, input_queue):

        model = load_or_create_model(self.model_name)

        features_board = np.zeros(((self.batch_size, 8, 8, 18)), np.int8)
        features_non_progress = np.zeros((self.batch_size, 1), np.float32)

        while True:
            response_queues = []
            for i in range(self.batch_size):
                request = input_queue.get()
                response_queues.append(request[0])
                features = request[1]
                features_board[i] = features[0]
                features_non_progress[i] = features[1]

            predictions = model.predict([features_board, features_non_progress])

            for i in range(self.batch_size):
                response_queues[i].put([predictions[0][i], predictions[1][i]])
