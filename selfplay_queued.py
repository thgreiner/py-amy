#!/usr/bin/env python3

from selfplay import selfplay
from queued_evaluator import QueuedEvaluator, MultiplexingEvaluator
from threading import Thread
from queue import Queue
import argparse
import chess.pgn
import time

class QueueSaver:
    def __init__(self, queue):
        self.queue = queue

    def __call__(self, game):
        self.queue.put(game)

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue

    def __call__(self):
        name = "LearnGames-{}.pgn".format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        while True:
            game = self.queue.get()
            with open(name, "a") as f:
                exporter = chess.pgn.FileExporter(f)
                game.accept(exporter)


if __name__ == "__main__":

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
