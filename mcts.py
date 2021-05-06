#!/usr/bin/env python3

import numpy as np

from kld import KLD
from move_selection import add_exploration_noise, add_bias_move
from tablebase import get_optimal_move
from non_blocking_console import NonBlockingConsole
from chess_input import Repr2D
from prometheus_client import Gauge
from mcts_stats import MCTS_Stats
from math import sqrt, log, exp

FORCED_PLAYOUT = 10000


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.turn = None
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


PB_C_INIT = 1.25
PB_C_BASE = 19652


def ucb_pb_c(parent_visit_count):
    pb_c = log((parent_visit_count + PB_C_BASE + 1) / PB_C_BASE)
    pb_c += PB_C_INIT
    pb_c *= sqrt(parent_visit_count)
    return pb_c


def ucb_score(parent_visit_count, child: Node):
    return child.value() + child.prior * ucb_pb_c(parent_visit_count) / (
        child.visit_count + 1
    )


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


# Select the child with the highest UCB score.
def select_child(node: Node):
    parent_visit_count = node.visit_count

    max_action = None
    max_ucb = None
    max_child = None

    k = ucb_pb_c(parent_visit_count)
    for action, child in node.children.items():
        u = child.value() + child.prior * k / (child.visit_count + 1)
        if max_ucb is None or u > max_ucb:
            max_action, max_ucb, max_child = action, u, child

    return max_action, max_child


def backpropagate(search_path, value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.turn != to_play else (1 - value)
        node.visit_count += 1


def is_singular_move(search_path, threshold):
    return len(search_path) > 1 and search_path[1].visit_count > threshold


class MCTS:
    def __init__(
        self,
        model,
        verbose=True,
        prefix=None,
        max_simulations=800,
        exploration_noise=True,
    ):
        self.model = model
        self.repr = Repr2D()

        self.verbose = verbose
        self.prefix = prefix
        self.max_simulations = max_simulations
        self.exploration_noise = exploration_noise

        self.best_move = None

        self.kldgain_stop = 0.0

    def set_kldgain_stop(self, kldgain):
        self.kldgain_stop = kldgain

    def model_name(self):
        return self.model.name
        # return f"{self.model.name}, {self.ucb_score}"

    def move_prob(self, logits, move, xor):
        sq = move.to_square ^ xor
        plane = self.repr.plane_index(move, xor)
        return exp(logits[sq, plane])

    def evaluate(self, node, board, full_check=False):

        node.turn = board.turn

        if full_check:
            if board.is_game_over(claim_draw=True):
                self.stats.observe_terminal_node()
                return score(board, board.result(claim_draw=True))
        else:
            # Consider any repetition a draw
            if (
                board.is_repetition(count=2)
                or board.is_insufficient_material()
                or board.is_fifty_moves()
            ):
                self.stats.observe_terminal_node()
                return 0.5

        legal_moves = list(board.generate_legal_moves())
        if len(legal_moves) == 0:
            self.stats.observe_terminal_node()
            return score(board, board.result(claim_draw=True))

        input_board = np.expand_dims(self.repr.board_to_array(board), axis=0)
        prediction = self.model.predict(input_board)

        value = (prediction[1].flatten())[0]
        # Transform [-1, 1] range to [0, 1]
        value = (value + 1) * 0.5

        logits = prediction[0].reshape(64, 73)

        xor = 0 if board.turn else 0x38

        # Check endgame tablebase
        tb_move, tb_value = get_optimal_move(board)

        if tb_value is not None and tb_value != "Draw":
            policy = {move: (1 if move == tb_move else 0) for move in legal_moves}
        else:
            policy = {move: (self.move_prob(logits, move, xor)) for move in legal_moves}

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

        return value

    def mcts(self, board, prefix, sample=True, limit=None, bias_move=None):
        self.stats = MCTS_Stats(self.model_name(), self.verbose, self.prefix)
        kld = KLD()

        root = Node(0)
        self.stats.observe_root_value(self.evaluate(root, board, full_check=True))

        if self.exploration_noise:
            add_exploration_noise(root)

        if bias_move:
            add_bias_move(root, bias_move)

        max_visit_count = self.max_simulations
        if limit is not None:
            max_visit_count = min(limit, max_visit_count)

        with NonBlockingConsole() as nbc:
            for iteration in range(max_visit_count):
                depth = 0

                node = root
                search_path = [node]
                while node.expanded():
                    move, node = select_child(node)
                    board.push(move)
                    search_path.append(node)
                    depth += 1

                value = self.evaluate(node, board, depth < 2)
                backpropagate(search_path, value, board.turn)

                self.stats.observe_depth(depth)

                for i in range(depth):
                    board.pop()

                if iteration > 0 and iteration % 100 == 0:
                    if self.verbose and iteration % 400 == 0:
                        self.stats.statistics(root, board)
                    kldgain = kld.update(root)

                    if kldgain is not None and kldgain < self.kldgain_stop:
                        break

                if root.visit_count >= max_visit_count:
                    break

                if is_singular_move(search_path, 4 * max_visit_count / 5):
                    break

                if nbc.get_data() == "\x1b":
                    break

        self.stats.statistics(root, board)

        white_win_prop = 1.0 - root.value() if board.turn else root.value()
        prop_gauge.labels(game=prefix).set(white_win_prop)

        return root

prop_gauge = Gauge("white_prob", "Win probability white", ["game"])
