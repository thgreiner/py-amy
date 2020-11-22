#!/usr/bin/env python3

from selfplay import selfplay
from pgn_writer import DefaultGameSaver
from queued_evaluator import QueuedEvaluator, MultiplexingEvaluator
from threading import Thread
from queue import Queue
import argparse
import time
from prometheus_client import start_http_server

class QueueSaver:
    def __init__(self, queue):
        self.queue = queue

    def __call__(self, game):
        self.queue.put(game)

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue

    def __call__(self):
        saver = DefaultGameSaver("LearnGames")
        while True:
            game = self.queue.get()
            saver(game)


if __name__ == "__main__":

    start_http_server(9100)

    parser = argparse.ArgumentParser(description="Execute parallel selfplay.")
    parser.add_argument('--model', help="model file name")
    args = parser.parse_args()

    multiplexing_evaluator = MultiplexingEvaluator(args.model, 20)
    input_queue = Queue()

    eval_thread = Thread(target = lambda: multiplexing_evaluator.run(input_queue))
    eval_thread.start()

    save_queue = Queue()
    saver = QueueSaver(save_queue)

    save_thread = Thread(target = QueueWriter(save_queue))
    save_thread.start()

    multiplexing_evaluator.model_loaded.wait()

    for i in range(24):
        qe = QueuedEvaluator(input_queue, multiplexing_evaluator.name)
        play_thread = Thread(target = lambda: selfplay(qe, 800, verbose=False, prefix=str(i), saver=saver))
        play_thread.start()
