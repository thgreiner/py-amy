from chess import Board
import chess.pgn
import random
import math
import numpy
import time

from datetime import date

from chess_input import Repr2D

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import click

from searcher import Searcher
import piece_square_eval

# This is required to load the model...
def my_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

model = load_model("combined-model.h5", custom_objects={'my_categorical_crossentropy': my_categorical_crossentropy})
repr = Repr2D()

C = 1.4

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
    return math.exp(logits[fr, plane])


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
    
    input_pos = repr.board_to_array(board).reshape(1, 8, 8, 17)
    # input_castling = repr.castling_to_array(board).reshape(1, 4)
    prediction = model.predict(input_pos)
    
    value = (prediction[1].flatten())[0]

    logits = prediction[0].reshape(64, 73)

    xor = 0 if board.turn else 0x38

    # Expand the node.
    node.turn = board.turn
    policy = {a: move_prob(logits, board, a, xor) for a in board.generate_legal_moves()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)

    return value


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0


def pv(board, node):
    stats = []
    for key, val in node.children.items():
        if val.visit_count > 0:
            stats.append((key, val.visit_count))

    if stats:
        stats = sorted(stats, key = lambda e: e[1], reverse=True)
        best_move = stats[0][0]
        line = board.san(best_move) + " "
        board.push(best_move)
        line += pv(board, node.children[best_move])
        board.pop()
        return line
    else:
        return ""

def statistics(root, board):
    global start_time, num_simulations, max_depth, sum_depth

    click.clear()
    print(board)

    print()
    print(pv(board, root))
    print()
    elapsed = time.perf_counter() - start_time
    print("{} simulations in {:.1f} seconds = {:.1f} simulations/sec".format(
        num_simulations,
        elapsed,
        num_simulations / elapsed
    ))
    print()
    avg_depth = sum_depth / num_simulations
    print("Max depth: {} Avg depth: {:.1f}".format(max_depth, avg_depth))
    print()

    best_move = None

    stats = []
    for key, val in root.children.items():
        if val.visit_count > 0:
            stats.append((board.san(key), val.value(), val.visit_count))

    stats = sorted(stats, key = lambda e: e[2], reverse=True)

    cnt = 0
    for s1 in stats:
        print("{:5s} {:5.1f}% {:5.0f} visits".format(
            s1[0], 100 * s1[1], s1[2]))
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
    pb_c_base = 900
    pb_c_init = 1.25

    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    
    # print("({:.2f} {:.2f}) ".format(prior_score, value_score), end='', flush=True)
    return prior_score + value_score

def backpropagate(search_path, value: float, to_play):
    for node in search_path:
        node.value_sum += (value if node.turn != to_play else (1 - value))
        node.visit_count += 1

def add_exploration_noise(node: Node):
    root_dirichlet_alpha = 0.3
    root_exploration_fraction = 0.2
    
    actions = node.children.keys()
    noise = numpy.random.gamma(root_dirichlet_alpha, 1, len(actions))
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

start_time = None
num_simulations = None
max_depth = None
sum_depth = None

def mcts(board):
    global start_time, num_simulations, max_depth, sum_depth
    
    start_time = time.perf_counter()
    num_simulations = 0
    max_depth = 0
    sum_depth = 0

    root = Node(0)
    evaluate(root, board)
    # add_exploration_noise(root)
    
    best_move = None
    for iteration in range(0, 800):
        num_simulations += 1
        depth = 0
        
        node = root
        search_path = [ node ]
        scratch_board = board.copy()
        while node.expanded():
            move, node = select_child(node)
            # print("{} ".format(scratch_board.san(move)), end='')
            scratch_board.push(move)
            search_path.append(node)
            depth += 1
        
        value = evaluate(node, scratch_board)
        # print("{:.1f}%        ".format(100 * value))
        backpropagate(search_path, value, scratch_board.turn)

        max_depth = max(max_depth, depth)
        sum_depth += depth
                
        if iteration % 100 == 0:
            statistics(root, board)

    best_move = statistics(root, board)
    return best_move

if __name__ == "__main__":
    # board, _ = Board.from_epd("4r2k/p5pp/8/3Q1b1q/2B2P1P/P1P2n2/5PK1/R6R b - -")
    game = chess.pgn.Game()
    game.headers["Event"] = "Test Game"
    game.headers["White"] = "Amy Zero"
    game.headers["Black"] = "Piece Square Tables"
    game.headers["Date"] = date.today().strftime("%Y.%m.%d")
    node = game

    board = Board()
    # board.set_fen("8/k7/5Q2/8/8/8/8/4K3 b - - 0 1")
    # opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5"
    # opening = "d4 d5"
    # opening = "e4 c5 Nf3 Nc6"
    opening = None
    if opening:
        for move in opening.split(" "):
            m = board.parse_san(move)
            node = node.add_variation(m)
            board.push(m)

    black_searcher = Searcher(lambda board: piece_square_eval.evaluate(board), "PieceSquareTables")

    while not board.is_game_over(claim_draw = True):
        print(board)
        if True or board.turn:
            best_move = mcts(board)
        else:
            best_move = board.san(black_searcher.select_move(board))
            
        print(best_move)
        m = board.parse_san(best_move)
        node = node.add_variation(m)
        board.push(m)

    game.headers["Result"] = board.result()

    with open("LearnGames.pgn", "a") as f:
        print(game, file=f, end="\n\n")
