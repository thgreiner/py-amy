#!/usr/bin/env python3

import chess
from chess import Board
import chess.pgn
from chess_input import Repr2D

import numpy as np
import sys
import time
import random
import argparse
from functools import partial

from threading import Thread
from queue import PriorityQueue

from prometheus_client import start_http_server, Counter, Gauge

from network import load_or_create_model, schedule_learn_rate

from dataclasses import dataclass, field
from typing import Any

# Checkpoint every "CHEKCPOINT" updates
CHECKPOINT = 100_000

# Maximum priority to assign an item in the position queue
MAX_PRIO = 1_000_000

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


class Node:

    def __init__(self):
        self.visit_count = 0
        self.learn_count = 0
        self.result = 0
        self.children = {}


def label_for_result(result, turn):
    if result == '1-0':
        if turn:
            return 1
        else:
            return -1
    if result == '0-1':
        if turn:
            return -1
        else:
            return 1

    return 0


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


def pos_generator(filename, elo_diff, skip_games, game_counter, queue):

    root = Node()

    with open(filename) as pgn:
        while True:
            skip_training = False

            try:
                game = chess.pgn.read_game(pgn)
            except UnicodeDecodeError or ValueError:
                continue
            if game is None:
                break

            result = game.headers["Result"]
            white = game.headers["White"]
            black = game.headers["Black"]
            date_of_game = game.headers["Date"]

            white_elo = game.headers.get("WhiteElo", "-")
            black_elo = game.headers.get("BlackElo", "-")

            if white_elo != "-" and black_elo != "-":
                w = int(white_elo)
                b = int(black_elo)
                if abs(w - b) < elo_diff:
                    # print("Skipping game - Elo diff less than {}}.".format(elo_diff))
                    continue
            elif elo_diff > 0:
                # print("Skipping game, one side has no Elo.")
                continue

            print("{} ({}) - {} ({}), {} {}        ". format(
                white, white_elo,
                black, black_elo,
                result, date_of_game), end='\r')

            if skip_games > 0:
                print("Skipping {} games...".format(skip_games), end='\r')
                skip_games -= 1
                skip_training = True

            game_counter.labels(result=result).inc()

            b = game.board()

            if game.headers.get("SetUp", "0") == "1":
                node = None
            else:
                node = root
            out_of_book = 0

            for move in game.mainline_moves():

                if node:
                    node.visit_count += 1
                    node.result += label_for_result(result, b.turn)

                if not skip_training:
                    train_data_board = repr.board_to_array(b)
                    train_data_moves = repr.legal_moves_mask(b)
                    train_data_non_progress = b.halfmove_clock / 100.0
                    train_labels1 = repr.move_to_array(b, move)
                    if node:
                        train_labels2 = node.result / node.visit_count
                    else:
                        train_labels2 = label_for_result(result, b.turn)

                    item = PrioritizedItem(
                        random.randint(0, MAX_PRIO),
                        ( train_data_board, 
                          train_data_moves, 
                          train_data_non_progress,
                          train_labels1,
                          train_labels2 ))
                    queue.put(item)

                if node:
                  if move in node.children:
                      node = node.children[move]
                  else:
                      out_of_book += 1
                      if out_of_book <= 10:
                          child_node = Node()
                          node.children[move] = child_node
                      else:
                          node = None

                b.push(move)
    queue.put(PrioritizedItem(MAX_PRIO, None))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training on a PGN file.")
    parser.add_argument("filename")
    parser.add_argument('--diff', type=int, help="minimum elo barrier", default=0)
    parser.add_argument('--model', help="model file name")
    parser.add_argument('--skip', type=int, help="games to skip", default=0)
    parser.add_argument('--test', action='store_const', const=True, default=False, help="run test instead of training")
    parser.add_argument('--batch_size', type=int, help="batch size", default=256)

    args = parser.parse_args()

    repr = Repr2D()

    model_name = args.model
    model = load_or_create_model(model_name)

    batch_size = args.batch_size
    train_data_board = np.zeros(((batch_size, 8, 8, repr.num_planes)), np.int8)
    train_data_moves = np.zeros((batch_size, 4672), np.int8)
    train_data_non_progress = np.zeros((batch_size, 1), np.float32)
    train_labels1 = np.zeros((batch_size, 4672), np.int8)
    train_labels2 = np.zeros((batch_size, 1), np.float32)

    start_time = time.perf_counter()

    start_http_server(9099)
    pos_counter = Counter('training_position_total', "Positions seen by training")
    batch_no_counter = Counter('training_batch_total', "Training batches")
    game_counter = Counter('training_game_total', "Games seen by training", [ "result" ])
    loss_gauge = Gauge('training_loss', "Training loss")
    moves_accuracy_gauge = Gauge('training_move_accuracy', "Move accuracy")
    score_mae_gauge = Gauge('training_score_mae', "Score mean absolute error")
    learn_rate_gauge = Gauge('training_learn_rate', "Learn rate")

    queue = PriorityQueue(maxsize = 50000)

    for iteration in range(100):

        cnt = 0
        samples = 0
        checkpoint_no = 0
        checkpoint_next = CHECKPOINT
        batch_no = 0

        pos_gen = partial(pos_generator, args.filename, args.diff, args.skip, game_counter, queue)

        t = Thread(target = pos_gen)
        t.start()

        while True:
            item = queue.get()
            sample = item.item
            
            if sample is None:
                break

            train_data_board[cnt] = sample[0]
            train_data_moves[cnt] = sample[1]
            train_data_non_progress[cnt, 0] = sample[2]
            train_labels1[cnt] = sample[3]
            train_labels2[cnt, 0] = sample[4]
            cnt += 1

            pos_counter.inc()

            if cnt >= batch_size:
                # print(train_labels2)
                train_data = [ train_data_board, train_data_moves, train_data_non_progress]
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


        # Train on the remainder of the dataset
        train_data = [ train_data_board[:cnt], train_data_moves[:cnt], train_data_non_progress[:cnt] ]
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
