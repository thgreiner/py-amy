#!/usr/bin/env python3

import argparse
import time
from functools import partial

from threading import Thread
from queue import PriorityQueue

from prometheus_client import start_http_server

from network import load_or_create_model

from pgn_reader import pos_generator
from train_loop import train_epoch


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training on a PGN file.")
    parser.add_argument("filename")
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

    batch_size = args.batch_size

    queue = PriorityQueue()

    for port in range(9099, 9104):
        try:
            start_http_server(port)
            print(f"Started http server on port {port}")
            break
        except OSError:
            pass

    for epoch in range(10):

        pos_gen = partial(pos_generator, args.filename, args.test, queue)

        t = Thread(target=pos_gen)
        t.start()

        if not args.test:
            wait_for_queue_to_fill(queue)

        train_epoch(model, args.batch_size, epoch, queue, args.test)

        if args.test:
            break

        if model_name is None:
            model.save("combined-model.h5")
        else:
            model.save(model_name)
            history_name = f"{model_name.removesuffix('.h5')}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.h5"
            model.save(f"model_history/{history_name}")
