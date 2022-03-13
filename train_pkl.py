#!/usr/bin/env python3

from chess_input import Repr2D

import time
import argparse

from random import shuffle, randint
from functools import partial

from threading import Thread
from queue import PriorityQueue

from prometheus_client import start_http_server

from network import load_or_create_model

from pgn_reader import end_of_input_item, randomize_item

from train_loop import train_epoch

import pickle

def wait_for_queue_to_fill(q):
    old_qsize = None
    for i in range(900):
        time.sleep(1)
        print("Waiting for queue to fill, current size is {}     ".format(q.qsize()))
        if q.qsize() > 50000:
            break
        if old_qsize is not None and old_qsize == q.qsize():
            break
        old_qsize = q.qsize()


def read_pickle(queue, test_mode):

    if test_mode:
        files = ["validation.pkl"]
        sample = 100
    else:
        files = [f"train-{i}.pkl" for i in range(10)]
        shuffle(files)
        sample = 5

    for filename in files:
        print(f"Reading {filename}")
        with open(f"data/{filename}", "rb") as fin:
            try:
                while True:
                    item = pickle.load(fin)
                    if randint(0, 99) < sample:
                        queue.put(randomize_item(item))
            except EOFError:
                pass

    queue.put(end_of_input_item())


def start_pos_gen_thread(queue, test_mode):
    pos_gen = partial(read_pickle, queue, test_mode)

    t = Thread(target=pos_gen)
    t.start()

    if not test_mode:
        wait_for_queue_to_fill(queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training on a PGN file.")
    parser.add_argument("--model", help="model file name")
    parser.add_argument(
        "--test",
        action="store_const",
        const=True,
        default=False,
        help="run test instead of training",
    )
    parser.add_argument("--batch_size", type=int, help="batch size", default=256)

    args = parser.parse_args()

    model_name = args.model

    model = load_or_create_model(model_name)
    model.summary()

    start_time = time.perf_counter()

    queue = PriorityQueue(maxsize=200000)

    start_http_server(9099)

    for epoch in range(20):

        start_pos_gen_thread(queue, args.test)

        train_epoch(model, args.batch_size, epoch, queue, args.test)

        if args.test:
            break

        if model_name is None:
            model.save("combined-model.h5")
        else:
            model.save(model_name)
            history_name = f"{model_name.removesuffix('.h5')}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.h5"
            model.save(f"model_history/{history_name}")
