#!/usr/bin/env python3

import chess.pgn
from chess_input import Repr2D

import random
import re

from prometheus_client import Counter

from dataclasses import dataclass, field
from typing import Any

# Maximum priority to assign an item in the position queue
MAX_PRIO = 1_000_000


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    data_board: Any = field(compare=False)
    data_non_progress: Any = field(compare=False)
    label_moves: Any = field(compare=False)
    label_value: Any = field(compare=False)
    label_result: Any = field(compare=False)


def label_for_result(result, turn):
    if result == "1-0":
        if turn:
            return [1, 0, 0]
        else:
            return [0, 1, 0]
    if result == "0-1":
        if turn:
            return [0, 1, 0]
        else:
            return [1, 0, 0]

    return [0, 0, 1]


repr = Repr2D()

re1 = re.compile("q=(.*); p=\[(.*)\]")
re2 = re.compile("(.*):(.*)")


def parse_mcts_result(input):
    m = re1.match(input)

    if m is None:
        return None, None

    q = float(m.group(1))

    variations = m.group(2).split(", ")

    v = {}
    for variation in variations:
        m2 = re2.match(variation)
        if m2 is not None:
            v[m2.group(1)] = float(m2.group(2))

    return q, v


def randomize_item(item):
    item.priority = random.randint(0, MAX_PRIO)
    return item


def traverse_game(node, board, queue, result, sample_rate, follow_variations=False):

    positions_created = 0

    if not follow_variations and not node.is_mainline():
        return positions_created

    move = node.move

    if node.comment and random.randint(0, 100) < sample_rate:

        q, policy = parse_mcts_result(node.comment)
        q = q * 2 - 1.0

        result_label = label_for_result(result, board.turn)
        # z = result_label[0] - result_label[1]

        train_data_board = repr.board_to_array(board)
        train_data_non_progress = board.halfmove_clock / 100.0
        train_labels1 = repr.policy_to_array(board, policy)

        item = PrioritizedItem(
            random.randint(0, MAX_PRIO),
            train_data_board,
            train_data_non_progress,
            train_labels1,
            q,
            result_label
        )
        queue.put(item)

        positions_created += 1

    if move is not None:
        board.push(move)

    for sibling in node.variations:
        positions_created += traverse_game(sibling, board, queue, result, sample_rate)

    if move is not None:
        board.pop()

    return positions_created


# Counter for monitoring no. of games
game_counter = Counter("training_game_total", "Games seen by training", ["result"])


def pos_generator(filename, test_mode, queue):

    sample_rate = 100 # if test_mode else 50

    cnt = 0
    with open(filename) as pgn:

        positions_created = 0
        while positions_created < 2500000:
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

            game_counter.labels(result=result).inc()

            cnt += 1
            print(
                "Parsing game #{} {}, {} positions".format(
                    cnt, date_of_game, positions_created
                ),
                end="\r",
            )

            positions_created += traverse_game(
                game, game.board(), queue, result, sample_rate
            )

    queue.put(PrioritizedItem(MAX_PRIO, None, None, None, None, None))
