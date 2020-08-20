#!/usr/bin/env python3

from chess import Board

from tablebase import get_optimal_move

import random
import math
import numpy as np
import time
import uuid
import textwrap
from non_blocking_console import NonBlockingConsole

from datetime import date

from chess_input import Repr2D

import click

from prometheus_client import Counter, Gauge

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


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


def add_exploration_noise(node: Node):
    root_dirichlet_alpha = 0.3
    root_exploration_fraction = 0.25

    actions = node.children.keys()
    noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
    noise /= np.sum(noise)
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# Select the child with the highest UCB score.
def select_child(node: Node):
    _, action, child = max(((ucb_score(node, child), action, child)
                           for action, child in node.children.items()),
                           key = lambda e: e[0])
    return action, child


def pv(board, node, variation=None):

    if variation == None:
        variation = []

    if len(node.children) == 0:
        return variation

    _, best_move = max(((child.visit_count, action)
                       for action, child in node.children.items()),
                       key = lambda e: e[0])

    variation.append(best_move)
    board.push(best_move)
    pv(board, node.children[best_move], variation)
    board.pop()
    return variation


# The score for a node is based on its value, plus an exploration bonus
# based on  the prior.
def ucb_score(parent: Node, child: Node):
    pb_c_base = 3000 #19652
    pb_c_init = 1.25

    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * max(child.prior, 0.02)
    value_score = child.value()

    return prior_score + value_score


def backpropagate(search_path, value: float, to_play):
    for node in search_path:
        node.value_sum += (value if node.turn != to_play else (1 - value))
        node.visit_count += 1


def sample_gumbel(a):
    b = [math.log(x) - math.log(-math.log(random.uniform(0, 1))) for x in a]
    return np.argmax(b)


def select_root_move(tree, move_count, sample=True):

    if len(tree.children) == 0:
        return None

    k = 2.0
    moves = []
    visits = []
    for key, val in tree.children.items():
        if val.visit_count > 0:
            moves.append(key)
            visits.append(val.visit_count ** k)

    if sample and move_count < 15:
        idx = sample_gumbel(visits)
    else:
        idx = np.argmax(visits)

    return moves[idx]


def is_singular_move(search_path, threshold):
    return len(search_path) >= 1 and search_path[1].visit_count > threshold


def variations(board, move, child, count):

    vars = []

    board.push(move)
    stats = [ (key, val)
        for key, val in child.children.items()
        if val.visit_count > 0 ]
    stats = sorted(stats, key = lambda e: e[1].visit_count, reverse=True)

    for m, grand_child in stats[:count]:
        line = [ m ]
        board.push(m)
        pv(board, grand_child, line)
        board.pop()

        vars.append(board.variation_san(line))

    board.pop()
    return vars


class MCTS:

    def __init__(self, model, verbose=True, prefix=None, max_simulations=800, exploration_noise=True):
        self.model = model
        self.repr = Repr2D()

        self.verbose = verbose
        self.prefix = prefix
        self.max_simulations = max_simulations
        self.exploration_noise = exploration_noise
        self.wrapper = wrapper = textwrap.TextWrapper(
            initial_indent = 9 * " ",
            subsequent_indent = 11 * " ",
            width=119)

        self.white_prop_gauge = Gauge('white_prob', "Win probability white")
        self.black_prop_gauge = Gauge('black_prob', "Win probability black")
        self.nps_gauge = Gauge('nps', 'Nodes per second')
        self.avg_depth_gauge = Gauge('avg_depth', 'Average search depth')
        self.median_depth_gauge = Gauge('median_depth', 'Median search depth')
        self.max_depth_gauge = Gauge('max_depth', 'Max search depth')
        self.terminal_node_gauge = Gauge('terminal_nodes', 'Percentage of terminal nodes encountered by search')
        self.best_move = None

    def move_prob(self, logits, move, xor):
        sq = move.to_square ^ xor
        plane = self.repr.plane_index(move, xor)
        return math.exp(logits[sq, plane])


    def evaluate(self, node, board):
        if board.is_game_over(claim_draw = True):
            self.terminal_nodes += 1
            winner = board.result(claim_draw = True)
            # print(winner)
            node.turn = board.turn
            return score(board, winner)

        input_board = self.repr.board_to_array(board).reshape(1, 8, 8, self.repr.num_planes)
        input_non_progress = np.array([ board.halfmove_clock / 100.0 ])

        prediction = self.model.predict([input_board, input_non_progress])

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
            policy = {move: (1 if move == tb_move else 0) for move in board.generate_legal_moves()}
        else:
            policy = {move: (self.move_prob(logits, move, xor)) for move in board.generate_legal_moves()}

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
            print(board)
            print()
            print(board.fen())
            # print()
            # print(board.variation_san(principal_variation))
            print()
            print("{} simulations in {:.1f} seconds = {:.1f} simulations/sec".format(
                self.num_simulations,
                elapsed,
                self.num_simulations / elapsed
            ))
            print()
            print("Max depth: {} Median depth: {} Avg depth: {:.1f} Terminal nodes: {:.1f}%".format(
                self.max_depth,
                self.median_depth,
                self.avg_depth,
                100 * self.terminal_nodes / self.num_simulations))
            print()

            stats = [ (key, val)
                for key, val in root.children.items()
                if val.visit_count > 0 ]
            stats = sorted(stats, key = lambda e: e[1].visit_count, reverse=True)

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
                print("{:5.1f}% {:10s} {:5.1f}% [{:4.1f}%] {:6d} visits {}".format(
                    100 * child_node.value(),
                    board.variation_san([move]),
                    100 * child_node.visit_count / self.num_simulations,
                    100 * child_node.prior,
                    child_node.visit_count,
                    message))
                message = ""
                if variations_cnt > 0:
                    variations_list = variations(board, move, child_node, variations_cnt)
                    for variation in variations_list:
                        for line in self.wrapper.wrap(variation):
                            print(line)
                    print()
                    variations_cnt -= 1

            if len(stats) > 10:
                print()
                remaining_moves = []
                for move, child_node in stats[10:]:
                    remaining_moves.append("{} ({:.1f}%, {})".format(
                        board.san(move),
                        100 * child_node.value(),
                        child_node.visit_count))
                for line in self.wrapper.wrap(", ".join(remaining_moves)):
                    print(line)

        else:
            print("{} - {}: {:4.1f}% {} [{:.1f} sims/s]".format(
                self.prefix,
                self.num_simulations,
                100 * root.children.get(principal_variation[0]).value(),
                board.variation_san(principal_variation),
                self.num_simulations / elapsed))

        self.nps_gauge.set(self.num_simulations / elapsed)
        self.terminal_node_gauge.set(100 * self.terminal_nodes / self.num_simulations)
        self.avg_depth_gauge.set(self.avg_depth)
        self.median_depth_gauge.set(self.median_depth)
        self.max_depth_gauge.set(self.max_depth)

    def mcts(self, board, sample=True):
        self.start_time = time.perf_counter()
        self.num_simulations = 0
        self.terminal_nodes = 0
        self.max_depth = 0
        self.sum_depth = 0
        self.depth_list = []

        root = Node(0)
        self.evaluate(root, board)

        if len(root.children) == 1:
            for best_move in root.children.keys():
                return best_move, root

        if self.exploration_noise:
            add_exploration_noise(root)

        best_move = None
        max_visit_count = self.max_simulations

        with NonBlockingConsole() as nbc:
            for iteration in range(max_visit_count):
                self.num_simulations += 1
                depth = 0

                node = root
                search_path = [ node ]
                while node.expanded():
                    move, node = select_child(node)
                    board.push(move)
                    search_path.append(node)
                    depth += 1

                value = self.evaluate(node, board)
                backpropagate(search_path, value, board.turn)

                self.max_depth = max(self.max_depth, depth)
                self.sum_depth += depth
                self.depth_list.append(depth)

                for i in range(depth):
                    board.pop()

                if iteration > 0 and iteration % 100 == 0:
                    self.statistics(root, board)

                if root.visit_count >= max_visit_count:
                    break

                if is_singular_move(search_path, 2 * max_visit_count / 3):
                    break

                if nbc.get_data() == '\x1b':
                    break

        self.statistics(root, board)

        selected_move = select_root_move(root, board.fullmove_number, sample)
        selected_move_child = root.children.get(selected_move)
        if board.turn:
            self.white_prop_gauge.set(selected_move_child.value())
        else:
            self.black_prop_gauge.set(selected_move_child.value())
        elapsed = time.perf_counter() - self.start_time

        return selected_move, root
