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

from pgn_reader import pos_generator

# Checkpoint every "CHEKCPOINT" updates
CHECKPOINT = 100_000

def stats(step_output):
    loss = step_output[0]
    moves_loss = step_output[1]
    score_loss = step_output[2]
    reg_loss = abs(loss - moves_loss - score_loss)

    moves_accuracy = step_output[3]
    score_mae = step_output[6]

    return "loss: {:.2f} = {:.2f} + {:.2f} + {:.2f}, move accuracy: {:4.1f}%, score mae: {:.2f}".format(
        loss,
        moves_loss, score_loss, reg_loss,
        moves_accuracy * 100, score_mae
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training on a PGN file.")
    parser.add_argument("filename")
    parser.add_argument('--diff', type=int, help="minimum elo diff", default=0)
    parser.add_argument('--model', help="model file name")
    parser.add_argument('--skip', type=int, help="games to skip", default=0)
    parser.add_argument('--test', action='store_const', const=True, default=False, help="run test instead of training")
    parser.add_argument('--batch_size', type=int, help="batch size", default=256)
    parser.add_argument('--min_elo', type=int, help='minimum elo threshold', default=0)

    args = parser.parse_args()

    model_name = args.model
    model = load_or_create_model(model_name)

    repr = Repr2D()

    batch_size = args.batch_size

    start_time = time.perf_counter()

    start_http_server(9099)
    pos_counter = Counter('training_position_total', "Positions seen by training")
    batch_no_counter = Counter('training_batch_total', "Training batches")
    game_counter = Counter('training_game_total', "Games seen by training", [ "result" ])
    loss_gauge = Gauge('training_loss', "Training loss")
    moves_accuracy_gauge = Gauge('training_move_accuracy', "Move accuracy")
    score_mae_gauge = Gauge('training_score_mae', "Score mean absolute error")
    learn_rate_gauge = Gauge('training_learn_rate', "Learn rate")
    qsize_gauge = Gauge("training_qsize", "Queue size")

    queue = PriorityQueue(maxsize = 400000)

    for iteration in range(100):

        train_data_board = np.zeros(((batch_size, 8, 8, repr.num_planes)), np.int8)
        train_data_non_progress = np.zeros((batch_size, 1), np.float32)
        train_labels1 = np.zeros((batch_size, 4672), np.float32)
        train_labels2 = np.zeros((batch_size, 1), np.float32)

        cnt = 0
        samples = 0
        checkpoint_no = 0
        checkpoint_next = CHECKPOINT
        batch_no = 0

        pos_gen = partial(pos_generator, args.filename, args.diff, args.min_elo,
                          args.skip, game_counter, queue)

        t = Thread(target = pos_gen)
        t.start()

        while True:
            
            item = queue.get()
            sample = item.item

            if sample is None:
                break

            pos_counter.inc()
            qsize_gauge.set(queue.qsize())

            train_data_board[cnt] = sample[0]
            train_data_non_progress[cnt, 0] = sample[1]
            train_labels1[cnt] = sample[2]
            train_labels2[cnt, 0] = sample[3]
            cnt += 1


            if cnt >= batch_size:
                # print(train_labels2)
                train_data = [ train_data_board, train_data_non_progress]
                train_labels = [ train_labels1, train_labels2 ]

                lr = schedule_learn_rate(model, batch_no)
                learn_rate_gauge.set(lr)
                batch_no += 1
                batch_no_counter.inc()

                if args.test:
                    results = model.test_on_batch(train_data, train_labels)
                else:
                    results = model.train_on_batch(train_data, train_labels)

                elapsed = time.perf_counter() - start_time

                samples += cnt
                print("{}.{}: {} in {:.1f}s".format(
                    iteration, samples, stats(results), elapsed))

                loss_gauge.set(results[0])
                moves_accuracy_gauge.set(results[3] * 100)
                score_mae_gauge.set(results[6])

                start_time = time.perf_counter()

                cnt = 0
                if samples >= checkpoint_next and not args.test:
                    checkpoint_no += 1
                    checkpoint_name = "checkpoint-{}.h5".format(checkpoint_no)
                    print("Checkpointing model to {}".format(checkpoint_name))
                    model.save(checkpoint_name)
                    checkpoint_next += CHECKPOINT


        # Train on the remainder of the dataset if enough data is remaining
        if cnt >= 16:
            train_data = [ train_data_board[:cnt], train_data_non_progress[:cnt] ]
            train_labels = [ train_labels1[:cnt], train_labels2[:cnt] ]

            start_time = time.perf_counter()
            if args.test:
                results = model.test_on_batch(train_data, train_labels)
            else:
                results = model.train_on_batch(train_data, train_labels)

            elapsed = time.perf_counter() - start_time

            samples += cnt
            print("{}.{}: {} in {:.1f}s]".format(
                iteration, samples, stats(results), elapsed))

        if not args.test:
            if model_name is None:
                model.save("combined-model.h5")
            else:
                model.save(model_name)
                
        batch_size *= 2
