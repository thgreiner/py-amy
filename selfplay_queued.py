#!/usr/bin/env python3

from selfplay import selfplay
from queued_evaluator import QueuedEvaluator, MultiplexingEvaluator
from threading import Thread
from queue import Queue

if __name__ == "__main__":

    model_name = "combined-model.h5"
    
    multiplexing_evaluator = MultiplexingEvaluator(model_name, 8)
    input_queue = Queue()

    eval_thread = Thread(target = lambda: multiplexing_evaluator.run(input_queue))
    eval_thread.start()

    for i in range(8):
        qe = QueuedEvaluator(input_queue)
        play_thread = Thread(target = lambda: selfplay(qe, False, i))
        play_thread.start()

