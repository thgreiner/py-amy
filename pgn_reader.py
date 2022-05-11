#!/usr/bin/env python3

import random
import re
from dataclasses import dataclass, field
from typing import Any

import chess.pgn
from prometheus_client import Counter

from chess_input import Repr2D

# Maximum priority to assign an item in the position queue
MAX_PRIO = 1_000_000


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    data_board: Any = field(compare=False)
    label_moves: Any = field(compare=False)
    label_value: Any = field(compare=False)
    label_wdl: Any = field(compare=False)
    label_moves_remaining: Any = field(compare=False)


def label_for_result(result, turn):
    if result == "1-0":
        if turn:
            return [1, 0, 0]
        else:
            return [0, 0, 1]
    if result == "0-1":
        if turn:
            return [0, 0, 1]
        else:
            return [1, 0, 0]

    return [0, 1, 0]


repr = Repr2D()

re1 = re.compile("q=(.*); p=\[(.*)\]")
re2 = re.compile("(.*):(.*)")

repetitions = 0


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


def traverse_game(game, board, queue, result, sample_rate):
    global repetitions

    positions_created = 0
    pos_map = dict()

    moves_remaining = len([x for x in game.mainline()])

    for node in game.mainline():

        move = node.move

        if node.comment:

            q, policy = parse_mcts_result(node.comment)
            q = q * 2 - 1.0
            z = label_for_result(result, board.turn)

            # q = 0.5 * (q + z[0] - z[2])

            train_data_board = repr.board_to_array(board)
            train_labels1 = repr.policy_to_array(board, policy)

            item = PrioritizedItem(
                random.randint(0, MAX_PRIO),
                train_data_board,
                train_labels1,
                q,
                z,
                moves_remaining,
            )

            moves_remaining -= 1

            key = board._transposition_key()
            if key in pos_map:
                repetitions += 1

            pos_map[key] = item

        if move is not None:
            board.push(move)

    for item in pos_map.values():
        if random.randint(0, 99) < sample_rate:
            queue.put(item)
            positions_created += 1

    return positions_created


# Counter for monitoring no. of games
game_counter = Counter("training_game_total", "Games seen by training", ["result"])


def pos_generator(filename, test_mode, queue):

    sample_rate = 100 if test_mode else 16

    cnt = 0
    with open(filename) as pgn:

        positions_created = 0
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

            game_counter.labels(result=result).inc()

            cnt += 1
            if cnt % 100 == 0:
                print(
                    "Parsing game #{} {}, {} positions (avg {:.1f} pos/game)".format(
                        cnt,
                        date_of_game,
                        positions_created,
                        positions_created / cnt,
                    ),
                    end="\r",
                )

            positions_created += traverse_game(
                game, game.board(), queue, result, sample_rate
            )

    print(
        f"Parsed {cnt} games, {positions_created} positions (avg {positions_created / cnt:.1f} pos/game)."
    )
    print(f"Repetitions suppressed: {repetitions}")

    queue.put(end_of_input_item())


def end_of_input_item():
    return PrioritizedItem(MAX_PRIO, None, None, None, None, None)
