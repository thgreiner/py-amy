#!/usr/bin/env python3

import numpy as np
import time

from chess_input import Repr2D
from prometheus_client import Counter, Gauge
from network import schedule_learn_rate
from train_stats import Stats

pos_counter = Counter("training_position_total", "Positions seen by training")
batch_no_counter = Counter("training_batch_total", "Training batches")
learn_rate_gauge = Gauge("training_learn_rate", "Learn rate")
qsize_gauge = Gauge("training_qsize", "Queue size")


def train_epoch(model, batch_size, epoch, queue, test_mode):
    
    start_time = time.perf_counter()
    stats = Stats()

    repr = Repr2D()
    train_data_board = np.zeros(((batch_size, 8, 8, repr.num_planes)), np.int8)
    train_labels1 = np.zeros((batch_size, 4672), np.float32)
    train_labels2 = np.zeros((batch_size, 1), np.float32)
    train_labels3 = np.zeros((batch_size, 3), np.float32)
    train_labels4 = np.zeros((batch_size, 1), np.float32)

    cnt = 0
    samples = 0
    batch_no = 0

    while True:

        item = queue.get()

        if item.data_board is None:
            break

        pos_counter.inc()
        qsize_gauge.set(queue.qsize())

        train_data_board[cnt] = item.data_board
        train_labels1[cnt] = item.label_moves.todense().reshape(4672)
        train_labels2[cnt, 0] = item.label_value
        train_labels3[cnt] = item.label_wdl
        train_labels4[cnt] = item.label_moves_remaining

        cnt += 1

        if cnt >= batch_size:
            # print(train_labels2)
            train_data = [train_data_board]
            train_labels = [train_labels1, train_labels2, train_labels3, train_labels4]

            if not test_mode:
                lr = schedule_learn_rate(model, epoch, batch_no)
                learn_rate_gauge.set(lr)

            batch_no += 1
            batch_no_counter.inc()

            if test_mode:
                results = model.test_on_batch(train_data, train_labels)
            else:
                results = model.train_on_batch(train_data, train_labels)

            elapsed = time.perf_counter() - start_time

            samples += cnt
            print(f"{epoch}.{samples}: {stats(results, cnt)} in {elapsed:.1f}s", end='\r')

            start_time = time.perf_counter()

            cnt = 0

    print()
    stats.write_to_file(model.name)
