#!/usr/bin/env python3

import chess.pgn
from chess_input import Repr2D

import random
import re

from dataclasses import dataclass, field
from typing import Any

# Maximum priority to assign an item in the position queue
MAX_PRIO = 1_000_000

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


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
    
def traverse_game(node, board, queue, skip_training, result, follow_variations=False):

    if not follow_variations and not node.is_mainline():
        return

    move = node.move

    if node.comment:

        q, policy = parse_mcts_result(node.comment)
        q = q * 2 - 1.0
        z = label_for_result(result, board.turn)

        if not skip_training:
            train_data_board = repr.board_to_array(board)
            train_data_non_progress = board.halfmove_clock / 100.0
            train_labels1 = repr.policy_to_array(board, policy)

            if node.is_mainline():
                train_labels2 = (q + z) * 0.5
            else:
                train_labels2 = q

            item = PrioritizedItem(
                random.randint(0, MAX_PRIO),
                ( train_data_board,
                  train_data_non_progress,
                  train_labels1,
                  train_labels2 ))
            queue.put(item)

    if move is not None:
        board.push(move)

    for sibling in node.variations:
        traverse_game(sibling, board, queue, skip_training, result)

    if move is not None:
        board.pop()


def pos_generator(filename, elo_diff, min_elo, skip_games, game_counter, queue):

    cnt = 0
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
                if min(w, b) < min_elo:
                    continue
            elif elo_diff > 0:
                # print("Skipping game, one side has no Elo.")
                continue

            if skip_games > 0:
                print("Skipping {} games...".format(skip_games), end='\r')
                skip_games -= 1
                skip_training = True

            game_counter.labels(result=result).inc()

            cnt += 1
            print("Parsing game #{} {}".format(cnt, date_of_game), end='\r')

            traverse_game(game, game.board(), queue, skip_training, result)

    queue.put(PrioritizedItem(MAX_PRIO, None))
