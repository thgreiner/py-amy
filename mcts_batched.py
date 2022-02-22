#!/usr/bin/env python3

import math
import numpy as np

from kld import KLD
from move_selection import add_exploration_noise
from tablebase import get_optimal_move
from non_blocking_console import NonBlockingConsole
from chess_input import Repr2D
from prometheus_client import Gauge
from mcts_stats import MCTS_Stats
from deferred_evaluator import DeferredEvaluator

BATCH_SIZE = 32


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.observed_count = 0
        self.turn = None
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.future_actions = None
        self.is_root = False

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def effective_count(self):
        return self.visit_count + self.observed_count


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
    target_set = [
        (action, child)
        for action, child in node.children.items()
        if child.expanded() or child.observed_count == 0
    ]

    if len(target_set) == 0:
        target_set = [(action, child) for action, child in node.children.items()]

    _, action, child = max(
        ((ucb_score(node, child), action, child) for action, child in target_set),
        key=lambda e: e[0],
    )
    return action, child


pb_c_base = 19625
pb_c_init = 1.25

# The score for a node is based on its value, plus an exploration bonus
# based on  the prior.
def ucb_score(parent: Node, child: Node):
    if parent.is_root:
        n_forced_playouts = math.sqrt(child.prior * parent.visit_count * 2)
        if child.effective_count() < n_forced_playouts:
            return 10000

    pb_c = math.log((parent.effective_count() + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.effective_count()) / (child.effective_count() + 1)

    prior_score = pb_c * child.prior  # max(child.prior, 0.02)
    value_score = child.value()

    return prior_score + value_score


def backpropagate(search_path, value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.turn != to_play else (1 - value)
        node.visit_count += 1
        node.observed_count -= 1


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
        self.deferred_evaluator = DeferredEvaluator(model, BATCH_SIZE)
        self.kldgain_stop = 0.0

    def set_kldgain_stop(self, kldgain):
        self.kldgain_stop = kldgain

    def move_prob(self, logits, move, xor):
        sq = move.to_square ^ xor
        plane = self.repr.plane_index(move, xor)
        return math.exp(logits[sq, plane])

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

        if tb_value is not None and tb_value != "Draw":
            policy = {move: (1 if move == tb_move else 0) for move in legal_moves}
        else:
            policy = {move: (self.move_prob(logits, move, xor)) for move in legal_moves}

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

        return value

    def evaluate_deferred_post(self, node, prediction, turn):

        value = (prediction[1].flatten())[0]
        # Transform [-1, 1] range to [0, 1]
        value = (value + 1) * 0.5

        logits = prediction[0].reshape(64, 73)

        xor = 0 if turn else 0x38

        # Expand the node.
        node.turn = turn

        policy = {
            move: (self.move_prob(logits, move, xor)) for move in node.future_actions
        }
        node.future_actions = None

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

        return value

    def evaluate_deferred(self, node, board):
        # Consider any repetition a draw
        if (
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
        self.deferred_evaluator.add(input_board)

        node.future_actions = legal_moves

        return None

    def mcts(self, board, prefix, sample=True, limit=None, bias_move=None):
        self.stats = MCTS_Stats(self.model.name, self.verbose)
        kld = KLD()

        root = Node(0)
        root.is_root = True
        self.stats.observe_root_value(self.evaluate(root, board, full_check=True))

        if self.exploration_noise:
            add_exploration_noise(root)

        if bias_move:
            add_bias_move(root, bias_move)

        max_visit_count = self.max_simulations
        if limit is not None:
            max_visit_count = min(limit, max_visit_count)

        num_simulations = 0
        next_statistics = 200
        next_check = 100

        with NonBlockingConsole() as nbc:
            for iteration in range(max_visit_count):
                to_evaluate = []
                turns = []

                self.deferred_evaluator.clear()

                for _ in range(BATCH_SIZE):
                    depth = 0
                    node = root
                    search_path = [node]
                    while node.expanded():
                        move, node = select_child(node)
                        board.push(move)
                        search_path.append(node)
                        depth += 1

                    if node.observed_count != 0:
                        # print("Detected observed final node at {}.".format(j), end='\r')
                        for i in range(depth):
                            board.pop()
                        break

                    for n in search_path:
                        n.observed_count += 1

                    value = self.evaluate_deferred(node, board)
                    if value is None:
                        to_evaluate.append(search_path)
                        turns.append(board.turn)
                    else:
                        backpropagate(search_path, value, board.turn)

                    self.stats.observe_depth(depth)
                    num_simulations += 1

                    for i in range(depth):
                        board.pop()

                for path, eval, turn in zip(
                    to_evaluate, self.deferred_evaluator.evaluate(), turns
                ):
                    value = self.evaluate_deferred_post(path[-1], eval, turn)
                    backpropagate(path, value, turn)

                if iteration > 0 and num_simulations > next_check:
                    if self.verbose and num_simulations > next_statistics:
                        self.stats.statistics(root, board)
                        next_statistics += 200

                    kldgain = kld.update(root)

                    if kldgain is not None and kldgain < self.kldgain_stop:
                        break
                    next_check += 100

                if root.visit_count >= max_visit_count:
                    break

                if is_singular_move(search_path, 2 * max_visit_count / 3):
                    break

                if nbc.get_data() == "\x1b":
                    break

        self.stats.statistics(root, board)

        white_win_prop = 1.0 - root.value() if board.turn else root.value()
        prop_gauge.labels(game=prefix).set(white_win_prop)

        return root

    def correct_forced_playouts(self, tree: Node):
        # No-op implementation
        return tree


prop_gauge = Gauge("white_prob", "Win probability white", ["game"])
