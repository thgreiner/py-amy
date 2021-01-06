#!/usr/bin/env python3

import math
import numpy as np

from ucb import FORCED_PLAYOUT, UCB
from kld import KLD
from move_selection import add_exploration_noise
from tablebase import get_optimal_move
from non_blocking_console import NonBlockingConsole
from chess_input import Repr2D
from prometheus_client import Gauge
from mcts_stats import MCTS_Stats


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.turn = None
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.is_root = False
        self.forced_playouts = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


# Select the child with the highest UCB score.
def select_child(node: Node, ucb_score):
    score, action, child = max(
        (
            (ucb_score(node, child, node.is_root), action, child)
            for action, child in node.children.items()
        ),
        key=lambda e: e[0],
    )

    if score == FORCED_PLAYOUT:
        child.forced_playouts += 1

    return action, child


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

        self.ucb_score = UCB(1.5)
        self.kldgain_stop = 0.0

    def set_pb_c_init(self, pb_c_init):
        self.ucb_score = UCB(pb_c_init)

    def set_kldgain_stop(self, kldgain):
        self.kldgain_stop = kldgain

    def model_name(self):
        return self.model.name
        # return f"{self.model.name}, {self.ucb_score}"

    def move_prob(self, logits, move, xor):
        sq = move.to_square ^ xor
        plane = self.repr.plane_index(move, xor)
        return math.exp(logits[sq, plane])

    def evaluate(self, node, board, check_draw=True):
        # Consider any repetition a draw
        if check_draw and (
            board.is_repetition(count=2)
            or board.is_insufficient_material()
            or board.is_fifty_moves()
        ):
            self.stats.observe_terminal_node()
            node.turn = board.turn
            return 0.5

        legal_moves = [move for move in board.generate_legal_moves()]
        if len(legal_moves) == 0:
            self.stats.observe_terminal_node()
            winner = board.result(claim_draw=True)
            # print(winner)
            node.turn = board.turn
            return score(board, winner)

        input_board = np.expand_dims(self.repr.board_to_array(board), axis=0)
        prediction = self.model.predict(input_board)

        value = (prediction[1].flatten())[0]
        # Transform [-1, 1] range to [0, 1]
        value = (value + 1) * 0.5

        logits = prediction[0].reshape(64, 73)

        xor = 0 if board.turn else 0x38

        # Check endgame tablebase
        tb_move, tb_value = get_optimal_move(board)

        # Expand the node.
        node.turn = board.turn

        if tb_value is not None and tb_value != "Draw":
            policy = {move: (1 if move == tb_move else 0) for move in legal_moves}
        else:
            policy = {move: (self.move_prob(logits, move, xor)) for move in legal_moves}

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

        return value

    def mcts(self, board, prefix, sample=True, limit=None):
        self.stats = MCTS_Stats(self.model_name(), self.verbose, self.prefix)
        kld = KLD()

        root = Node(0)
        root.is_root = True
        self.evaluate(root, board, check_draw=False)

        if self.exploration_noise:
            add_exploration_noise(root)

        max_visit_count = self.max_simulations
        if limit is not None:
            max_visit_count = min(limit, max_visit_count)

        with NonBlockingConsole() as nbc:
            for iteration in range(max_visit_count):
                depth = 0

                node = root
                search_path = [node]
                while node.expanded():
                    move, node = select_child(node, self.ucb_score)
                    board.push(move)
                    search_path.append(node)
                    depth += 1

                value = self.evaluate(node, board)
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

    def correct_forced_playouts(self, tree: Node):

        _, best_move = max(
            ((child.visit_count, action) for action, child in tree.children.items()),
            key=lambda e: e[0],
        )

        best_ucb_score = self.ucb_score(tree, tree.children[best_move])

        for action, child in tree.children.items():
            if action == best_move:
                continue

            actual_playouts = child.visit_count

            for i in range(1, child.forced_playouts + 1):
                child.visit_count = actual_playouts - i
                tmp_ucb_score = self.ucb_score(tree, tree.children[best_move])
                if tmp_ucb_score > best_ucb_score:
                    child.visit_count = actual_playouts - i + 1
                    break

        return tree


prop_gauge = Gauge("white_prob", "Win probability white", ["game"])
