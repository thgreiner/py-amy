from chess import Board
import chess.pgn
import random
import math
import numpy as np
import time

from datetime import date

from chess_input import Repr2D

import click

from searcher import Searcher, AmySearcher
import piece_square_eval
from pos_generator import generate_kxk

from network import load_or_create_model

MAX_HALFMOVES_IN_GAME = 200

# For KQK training
# MAX_HALFMOVES_IN_GAME = 60

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


def move_prob(logits, board, move, xor):
    fr = move.from_square ^ xor
    plane = repr.plane_index(move, xor)
    return logits[fr, plane]


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


def evaluate(node, board):
    if board.is_game_over(claim_draw = True):
        winner = board.result(claim_draw = True)
        # print(winner)
        node.turn = board.turn
        return score(board, winner)

    input_board = repr.board_to_array(board).reshape(1, 8, 8, repr.num_planes)
    input_moves = repr.legal_moves_mask(board).reshape(1, 4672)
    prediction = model.predict([input_board, input_moves])

    value = (prediction[1].flatten())[0]
    # Transform [-1, 1] range to [0, 1]
    value = (value + 1) * 0.5

    logits = prediction[0].reshape(64, 73)

    xor = 0 if board.turn else 0x38

    # Expand the node.
    node.turn = board.turn
    policy = {a: move_prob(logits, board, a, xor) for a in board.generate_legal_moves()}
    # We don't need to normalize - softmax does this for us
    # policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p)

    return value


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


def pv(board, node, variation):

    best_move = None
    best_visits = 0

    for key, val in node.children.items():
        if val.visit_count > 0:
            if best_move is None or val.visit_count > best_visits:
                best_move = key
                best_visits = val.visit_count

    if best_move is None:
        return

    variation.append(best_move)
    board.push(best_move)
    pv(board, node.children[best_move], variation)
    board.pop()


def statistics(root, board):
    global start_time, num_simulations, max_depth, sum_depth, depth_list

    click.clear()
    print(board)
    print()
    print(board.fen())

    print()
    principal_variation = []
    pv(board, root, principal_variation)
    print(board.variation_san(principal_variation))
    print()
    elapsed = time.perf_counter() - start_time
    print("{} simulations in {:.1f} seconds = {:.1f} simulations/sec".format(
        num_simulations,
        elapsed,
        num_simulations / elapsed
    ))
    print()

    avg_depth = sum_depth / num_simulations
    tmp = np.array(depth_list)
    median_depth = np.median(tmp, overwrite_input=True)
    print("Max depth: {} Median depth: {} Avg depth: {:.1f}".format(
        max_depth, median_depth, avg_depth))
    print()

    best_move = None

    stats = []
    for key, val in root.children.items():
        if val.visit_count > 0:
            stats.append((board.san(key), val.value(), val.visit_count, val.prior))

    stats = sorted(stats, key = lambda e: e[2], reverse=True)

    cnt = 0
    for s1 in stats:
        print("{:5s} {:5.1f}% {:5.0f} visits  [{:4.1f}%]".format(
            s1[0], 100 * s1[1], s1[2], 100 * s1[3]))
        cnt += 1
        if cnt >= 10:
            break

    return stats[0][0]

# Select the child with the highest UCB score.
def select_child(node: Node):
    _, action, child = max(((ucb_score(node, child), action, child)
                           for action, child in node.children.items()),
                           key = lambda e: e[0])
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node):
    pb_c_base = 19652
    pb_c_init = 3.5 # 1.25

    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()

    return prior_score + value_score

def backpropagate(search_path, value: float, to_play):
    for node in search_path:
        node.value_sum += (value if node.turn != to_play else (1 - value))
        node.visit_count += 1

def add_exploration_noise(node: Node):
    root_dirichlet_alpha = 0.3
    root_exploration_fraction = 0.2

    actions = node.children.keys()
    noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

start_time = None
num_simulations = None
max_depth = None
sum_depth = None
depth_list = None

def mcts(board):
    global start_time, num_simulations, max_depth, sum_depth, depth_list

    start_time = time.perf_counter()
    num_simulations = 0
    max_depth = 0
    sum_depth = 0
    depth_list = []

    root = Node(0)
    evaluate(root, board)

    if len(root.children) == 1:
        for best_move in root.children.keys():
            return board.san(best_move)

    # add_exploration_noise(root)

    best_move = None
    for iteration in range(80000):
        num_simulations += 1
        depth = 0

        node = root
        search_path = [ node ]
        scratch_board = board.copy()
        while node.expanded():
            move, node = select_child(node)
            scratch_board.push(move)
            search_path.append(node)
            depth += 1

        value = evaluate(node, scratch_board)
        backpropagate(search_path, value, scratch_board.turn)

        max_depth = max(max_depth, depth)
        sum_depth += depth
        depth_list.append(depth)

        if iteration % 100 == 0:
            statistics(root, board)

    best_move = statistics(root, board)
    return best_move

if __name__ == "__main__":

    model = load_or_create_model("combined-model.h5")
    repr = Repr2D()

    total_positions = 0
    while total_positions < 4096:
        # board, _ = Board.from_epd("4r2k/p5pp/8/3Q1b1q/2B2P1P/P1P2n2/5PK1/R6R b - -")

        board = Board()
        # board = generate_kxk()
        # board.set_fen("8/k7/5Q2/8/8/8/8/4K3 b - - 0 1")

        opening = None
        opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5"
        opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5 Bd3 h6 Bf4 b4 Ne4 Nxe4 Bxe4 Ba6 Qa4 Bb5"
        # opening = "d4 d5"
        # opening = "e4 c5 Nf3 Nc6"
        if opening:
            for move in opening.split(" "):
                m = board.parse_san(move)
                board.push(m)

        black_searcher = Searcher(lambda board: piece_square_eval.evaluate(board), "PieceSquareTables")
        amy_searcher = AmySearcher()

        while not board.is_game_over(claim_draw = True) and board.halfmove_clock < MAX_HALFMOVES_IN_GAME:
            if board.turn:
                best_move = mcts(board)
                # best_move = board.san(amy_searcher.select_move(board))
            else:
                # best_move = mcts(board)
                best_move = board.san(black_searcher.select_move(board))
            m = board.parse_san(best_move)
            board.push(m)
            total_positions += 1

        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "Test Game"
        game.headers["White"] = "Amy 0.9.1"
        game.headers["Black"] = "Amy Zero"
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        game.headers["Result"] = board.result()

        with open("LearnGames.pgn", "a") as f:
            print(game, file=f, end="\n\n")
