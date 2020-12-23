#!/usr/bin/env python3

from ucb import FORCED_PLAYOUT, UCB
from pv import variations
from kld import KLD
from move_selection import select_root_move, add_exploration_noise

from tablebase import get_optimal_move

import math
import numpy as np
import time
import textwrap
from non_blocking_console import NonBlockingConsole

from chess_input import Repr2D

import click

from prometheus_client import Counter, Gauge, Histogram

from colors import color


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
    return len(search_path) >= 1 and search_path[1].visit_count > threshold


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
        self.wrapper = textwrap.TextWrapper(
            initial_indent=9 * " ", subsequent_indent=11 * " ", width=119
        )

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

    def evaluate(self, node, board):
        if board.is_game_over(claim_draw=True):
            self.terminal_nodes += 1
            terminal_nodes_counter.inc()
            winner = board.result(claim_draw=True)
            # print(winner)
            node.turn = board.turn
            return score(board, winner)

        input_board = self.repr.board_to_array(board).reshape(
            1, 8, 8, self.repr.num_planes
        )

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
            policy = {
                move: (1 if move == tb_move else 0)
                for move in board.generate_legal_moves()
            }
        else:
            policy = {
                move: (self.move_prob(logits, move, xor))
                for move in board.generate_legal_moves()
            }

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

        return value

    def statistics(self, root, board):

        elapsed = time.perf_counter() - self.start_time

        if self.verbose:
            self.avg_depth = self.sum_depth / self.num_simulations
            tmp = np.array(self.depth_list)
            self.median_depth = np.median(tmp, overwrite_input=True)

            click.clear()

            print(
                f"{board}   {'White' if board.turn else 'Black'}: {self.model_name()}"
            )
            print()
            print(board.fen())
            # print()
            # print(board.variation_san(principal_variation))
            print()
            print(
                "{} simulations in {:.1f} seconds = {:.1f} simulations/sec".format(
                    self.num_simulations, elapsed, self.num_simulations / elapsed
                )
            )
            print()
            print(
                "Max depth: {} Median depth: {} Avg depth: {:.1f} Terminal nodes: {:.1f}%".format(
                    self.max_depth,
                    self.median_depth,
                    self.avg_depth,
                    100 * self.terminal_nodes / self.num_simulations,
                )
            )
            print()

            stats = [
                (key, val) for key, val in root.children.items() if val.visit_count > 0
            ]
            stats = sorted(stats, key=lambda e: e[1].visit_count, reverse=True)

            variations_cnt = 3
            print(" Score Line      Visit-% [prior]")
            print()

            if len(stats) > 0:
                current_best_move = stats[0][0]
                if self.best_move is None or self.best_move != current_best_move:
                    self.best_move = current_best_move
                    self.best_move_found = self.num_simulations

            message = ""
            if self.best_move is not None:
                message = "   (since iteration {})".format(self.best_move_found)

            for move, child_node in stats[:10]:
                print(
                    color(
                        "{:5.1f}% {:10s} {:5.1f}% [{:4.1f}%] {:6d} visits {}".format(
                            100 * child_node.value(),
                            board.variation_san([move]),
                            100 * child_node.visit_count / self.num_simulations,
                            100 * child_node.prior,
                            child_node.visit_count,
                            message,
                        ),
                        get_color(child_node.value()),
                    )
                )
                message = ""
                if variations_cnt > 0:
                    variations_list = variations(
                        board, move, child_node, variations_cnt
                    )
                    for variation in variations_list:
                        for line in self.wrapper.wrap(variation):
                            print(line)
                    print()
                    variations_cnt -= 1

            if len(stats) > 10:
                print()
                remaining_moves = []
                for move, child_node in stats[10:]:
                    remaining_moves.append(
                        color(
                            "{} ({:.1f}%, {})".format(
                                board.san(move),
                                100 * child_node.value(),
                                child_node.visit_count,
                            ),
                            get_color(child_node.value()),
                        )
                    )
                print(", ".join(remaining_moves))

        else:
            best_move = select_root_move(root, board.fullmove_number, False)
            variations_list = variations(board, best_move, root.children[best_move], 1)
            if len(variations_list) == 0:
                variations_list.append("")

            print(
                color(
                    "{:3} - {:4} {:5.1f}% {:12} [{:.1f} sims/s]  {}".format(
                        self.prefix or "",
                        self.num_simulations,
                        100 * root.children[best_move].value(),
                        board.variation_san([best_move]),
                        self.num_simulations / elapsed,
                        variations_list[0],
                    ),
                    get_color(root.children[best_move].value()),
                )
            )

    def mcts(self, board, prefix, sample=True, limit=None):
        self.start_time = time.perf_counter()
        self.num_simulations = 0
        self.terminal_nodes = 0
        self.max_depth = 0
        self.sum_depth = 0
        self.depth_list = []
        self.kld = KLD()

        root = Node(0)
        root.is_root = True
        self.evaluate(root, board)

        if self.exploration_noise:
            add_exploration_noise(root)

        max_visit_count = self.max_simulations
        if limit is not None:
            max_visit_count = min(limit, max_visit_count)

        with NonBlockingConsole() as nbc:
            for iteration in range(max_visit_count):
                self.num_simulations += 1
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

                depth_histogram.observe(depth)
                self.max_depth = max(self.max_depth, depth)
                self.sum_depth += depth
                self.depth_list.append(depth)

                for i in range(depth):
                    board.pop()

                nodes_counter.inc()

                if iteration > 0 and iteration % 100 == 0:
                    if self.verbose and iteration % 400 == 0:
                        self.statistics(root, board)
                    kldgain = self.kld.update(root)

                    if kldgain is not None and kldgain < self.kldgain_stop:
                        break


                if root.visit_count >= max_visit_count:
                    break

                if is_singular_move(search_path, 4 * max_visit_count / 5):
                    break

                if nbc.get_data() == "\x1b":
                    break

        self.statistics(root, board)

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
nodes_counter = Counter("nodes", "Nodes visited")
terminal_nodes_counter = Counter("terminal_nodes", "Terminal nodes visited")
depth_histogram = Histogram(
    "depth",
    "Search depth",
    buckets=[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20,
        24,
        28,
        32,
        36,
        40,
        48,
        56,
        64,
    ],
)


def get_color(x):
    t = min(int(x * 13), 12)
    if t < 6:
        return 16 + 5 * 36 + 6 * t + t
    elif t > 6:
        t = 12 - t
        return 16 + 36 * t + 6 * 5 + t
    else:
        return 0
