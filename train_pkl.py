#!/usr/bin/env python3

from chess_input import Repr2D

import numpy as np
import time
import argparse

from random import shuffle, randint
from functools import partial

from threading import Thread
from queue import PriorityQueue

from prometheus_client import start_http_server, Counter, Gauge

from network import load_or_create_model, schedule_learn_rate

from pgn_reader import end_of_input_item, randomize_item

from train_stats import Stats

import pickle

# Checkpoint every "CHEKCPOINT" updates
CHECKPOINT = 100_000


def wait_for_queue_to_fill(q):
    old_qsize = None
    for i in range(900):
        time.sleep(1)
        print("Waiting for queue to fill, current size is {}     ".format(q.qsize()))
        if q.qsize() > 10000:
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
        sample = 10

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

    repr = Repr2D()

    batch_size = args.batch_size

    start_time = time.perf_counter()

    pos_counter = Counter("training_position_total", "Positions seen by training")
    batch_no_counter = Counter("training_batch_total", "Training batches")
    loss_gauge = Gauge("training_loss", "Training loss")
    moves_accuracy_gauge = Gauge("training_move_accuracy", "Move accuracy")
    moves_top5_accuracy_gauge = Gauge(
        "training_move_top5_accuracy", "Top 5 move accuracy"
    )
    score_mae_gauge = Gauge("training_score_mae", "Score mean absolute error")
    learn_rate_gauge = Gauge("training_learn_rate", "Learn rate")
    qsize_gauge = Gauge("training_qsize", "Queue size")

    queue = PriorityQueue(maxsize=200000)

    start_http_server(9099)

    for iteration in range(100):

        stats = Stats()

        train_data_board = np.zeros(((batch_size, 8, 8, repr.num_planes)), np.int8)
        train_labels1 = np.zeros((batch_size, 4672), np.float32)
        train_labels2 = np.zeros((batch_size, 1), np.float32)

        cnt = 0
        samples = 0
        checkpoint_no = 0
        checkpoint_next = CHECKPOINT
        batch_no = 0

        start_pos_gen_thread(queue, args.test)

        while True:

            item = queue.get()

            if item.data_board is None:
                break

            pos_counter.inc()
            qsize_gauge.set(queue.qsize())

            train_data_board[cnt] = item.data_board
            train_labels1[cnt] = item.label_moves.todense().reshape(4672)
            train_labels2[cnt, 0] = item.label_value
            cnt += 1

            if cnt >= batch_size:
                # print(train_labels2)
                train_data = [train_data_board]
                train_labels = [train_labels1, train_labels2]

                lr = schedule_learn_rate(model, iteration, batch_no)
                learn_rate_gauge.set(lr)
                batch_no += 1
                batch_no_counter.inc()

                if args.test:
                    results = model.test_on_batch(train_data, train_labels)
                else:
                    results = model.train_on_batch(train_data, train_labels)

                elapsed = time.perf_counter() - start_time

                samples += cnt
                print(f"{iteration}.{samples}: {stats(results, cnt)} in {elapsed:.1f}s")

                loss_gauge.set(results[0])
                moves_accuracy_gauge.set(results[3] * 100)
                moves_top5_accuracy_gauge.set(results[4] * 100)
                score_mae_gauge.set(results[5])

                start_time = time.perf_counter()

                cnt = 0
                if samples >= checkpoint_next and not args.test:
                    checkpoint_no += 1
                    checkpoint_name = f"checkpoint-{checkpoint_no}.h5"
                    print(f"Checkpointing model to {checkpoint_name}")
                    model.save(checkpoint_name)
                    checkpoint_next += CHECKPOINT

        stats.write_to_file(model.name)

        if args.test:
            break

        if model_name is None:
            model.save("combined-model.h5")
        else:
            model.save(model_name)

        # Every 2 iterations, double the batch size
        # if iteration % 2 == 1:
        #     batch_size *= 2
