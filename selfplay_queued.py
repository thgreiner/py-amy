#!/usr/bin/env python3

from selfplay import selfplay
from queued_evaluator import QueuedEvaluator, MultiplexingEvaluator
from threading import Thread
from queue import Queue
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Execute parallel selfplay.")
    parser.add_argument('--model', help="model file name")
    args = parser.parse_args()

    multiplexing_evaluator = MultiplexingEvaluator(args.model, 20)
    input_queue = Queue()

    eval_thread = Thread(target = lambda: multiplexing_evaluator.run(input_queue))
    eval_thread.start()

    multiplexing_evaluator.model_loaded.wait()

    for i in range(24):
        qe = QueuedEvaluator(input_queue, multiplexing_evaluator.name)
        play_thread = Thread(target = lambda: selfplay(qe, 800, verbose=False, prefix=str(i)))
        play_thread.start()
