#!/usr/bin/env python3

import chess
from chess import Board
import chess.pgn

from chess_input import Repr2D

import numpy as np
import sys
import time
import argparse
from functools import partial

from threading import Thread
from queue import PriorityQueue

from prometheus_client import start_http_server, Counter, Gauge

from network import load_or_create_model, schedule_learn_rate

from pgn_reader import pos_generator, randomize_item

from train_stats import Stats

# Checkpoint every "CHEKCPOINT" updates
CHECKPOINT = 100_000


def wait_for_queue_to_fill(q):
    old_qsize = None
    for i in range(900):
        time.sleep(1)
        print("Waiting for queue to fill, current size is {}     ".format(q.qsize()))
        if q.qsize() > 100000:
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

    queue = PriorityQueue()
    queue2 = PriorityQueue()

    pos_gen = partial(pos_generator, args.filename, args.test, queue)

    t = Thread(target=pos_gen)
    t.start()

    if not args.test:
        wait_for_queue_to_fill(queue)

    start_http_server(9099)

    for iteration in range(6):

        stats = Stats()

        train_data_board = np.zeros(((batch_size, 8, 8, repr.num_planes)), np.int8)
        train_labels1 = np.zeros((batch_size, 4672), np.float32)
        train_labels2 = np.zeros((batch_size, 1), np.float32)

        cnt = 0
        samples = 0
        checkpoint_no = 0
        checkpoint_next = CHECKPOINT
        batch_no = 0

        while True:

            item = queue.get()

            if item.data_board is None:
                queue2.put(item)
                break

            queue2.put(randomize_item(item))

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
                print(
                    "{}.{}: {} in {:.1f}s".format(
                        iteration, samples, stats(results, cnt), elapsed
                    )
                )

                loss_gauge.set(results[0])
                moves_accuracy_gauge.set(results[3] * 100)
                moves_top5_accuracy_gauge.set(results[4] * 100)
                score_mae_gauge.set(results[5])

                start_time = time.perf_counter()

                cnt = 0
                if samples >= checkpoint_next and not args.test:
                    checkpoint_no += 1
                    checkpoint_name = "checkpoint-{}.h5".format(checkpoint_no)
                    print("Checkpointing model to {}".format(checkpoint_name))
                    model.save(checkpoint_name)
                    checkpoint_next += CHECKPOINT

        stats.write_to_file(model.name)

        if args.test:
            break

        if model_name is None:
            model.save("combined-model.h5")
        else:
            model.save(model_name)

        queue, queue2 = queue2, queue

        # Every 2 iterations, double the batch size
        if iteration % 2 == 1:
            batch_size *= 2
