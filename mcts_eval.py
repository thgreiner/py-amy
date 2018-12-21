from chess import Board
import random
from math import sqrt
from math import log

from chess_input import Repr2D

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import click

model = load_model("move-model.h5")
repr = Repr2D()

score_model = load_model("score-model.h5")

C = 1.4

def sort_key(from_pred, to_pred, board, move, xor):
    type = board.piece_at(move.from_square).piece_type
    fr = move.from_square ^ xor
    to = move.to_square ^ xor
    return from_pred[fr] * to_pred[type-1][to]


def choose_move(board, model, moves=None):
    input = repr.board_to_array(board)
    predictions = model.predict([input.reshape(1, 8, 8, 12)])
    from_pred = predictions[:,:,:,0].flatten()
    to_pred   = [ predictions[:,:,:,i].flatten() for i in range(1,7)]

    xor = 0
    if not board.turn:
        xor = 0x38

    if moves is None:
        moves = list(board.generate_legal_moves())

    moves = [(m, sort_key(from_pred, to_pred, board, m, xor)) for m in moves]
    total = sum([m[1] for m in moves])
    moves = [(m[0], 100 * m[1] / total) for m in moves]
    moves = sorted(moves, key=lambda m: m[1], reverse=True)

    # cnt = 0
    # for m in moves:
    #     print("{:5s} {:5.1f}%".format(board.san(m[0]), m[1]))
    #     cnt += 1
    #     if cnt >= 5:
    #         break

    k = random.uniform(0, 100)
    best_move = moves[0][0]
    for m in moves:
        p = m[1]
        if k < p:
            best_move = m[0]
            break
        else:
            k -= p

    return best_move


def evaluate(board):
    input_pos = repr.board_to_array(board).reshape(1, 8, 8, 12)
    input_castling = repr.castling_to_array(board).reshape(1, 4)
    prediction = score_model.predict([input_pos, input_castling]).flatten()
    return (1 + prediction[0]) * .5


def playout(board):
    if board.is_game_over(claim_draw = True):
        return 1.0 - score(board, board.result(claim_draw = True))

    return 1.0 - evaluate(board)


def score(board, winner):
    if board.turn and (winner == "1-0"):
        return 1
    if (not board.turn) and (winner == "0-1"):
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0

def select_move(board, node):
    if board.is_game_over(claim_draw = True):
        winner = board.result(claim_draw = True)
        # print(winner)
        return 1.0 - score(board, winner)

    moves = list(board.generate_legal_moves())

    visited = list()
    non_visited = list()

    for m in moves:
        san = board.san(m)
        if san in node:
            visited.append(m)
        else:
            non_visited.append(m)

    # print("visited: {}".format(visited))
    # print("non_visited: {}".format(non_visited))

    if non_visited:
        m = choose_move(board, model, non_visited)
        # print("{} [".format(board.san(m)), end='')
        board.push(m)
        winner = playout(board)
        board.pop()

        # print("] {}".format(winner))

        d = { "plays": 1, "wins": 0}
        node[board.san(m)] = d
        node["plays"] += 1
        # node["wins"] += score(board, winner)

        return 1.0 - winner

    else:
        visit_count = node["plays"]
        # print("Visit count: {}".format(visit_count))

        selected_move = None
        selected_prob = None
        selected_child_node = None

        for m in moves:
            san = board.san(m)
            child = node[san]

            child_wins = child["wins"]
            child_plays = child["plays"]

            child_prob = child_wins / child_plays + C * sqrt(log(visit_count) / child_plays)
            # print("{}: plays:{} wins:{} {}".format(san, child_plays, child_wins, child_prob))

            if selected_move is None or child_prob > selected_prob:
                selected_move = m
                selected_prob = child_prob
                selected_child_node = child

        # print("{} ".format(board.san(selected_move)), end='')

        board.push(selected_move)
        winner = select_move(board, selected_child_node)
        board.pop()

        # print("winner: {}".format(winner))

        node["plays"] += 1
        selected_child_node["wins"] += winner

        return 1.0 - winner


def pv(board, node):
    stats = []
    for key, val in node.items():
        if isinstance(val, dict):
            stats.append((key, val["plays"]))

    if stats:
        stats = sorted(stats, key = lambda e: e[1], reverse=True)
        best_move = stats[0][0]
        move = board.parse_san(best_move)
        board.push(move)
        line = best_move + " " + pv(board, node[best_move])
        board.pop()
        return line
    else:
        return ""

def statistics(node):
    best_move = None
    best_visits = None
    best_wins = None

    stats1 = []
    stats2 = []
    for key, val in node.items():
        if isinstance(val, dict):
            win_ratio = val["wins"] / val["plays"]
            stats1.append((key, win_ratio, val["plays"]))
            stats2.append((key, val["plays"]))

    stats1 = sorted(stats1, key = lambda e: e[1], reverse=True)
    stats2 = sorted(stats2, key = lambda e: e[1], reverse=True)
    cnt = 0
    for s1, s2 in zip(stats1, stats2):
        print("{:5s} {:5.1f}% {:5.0f} visits   {:5s} {:5.0f} visits".format(
            s1[0], 100 * s1[1], s1[2],
            s2[0], s2[1]))
        cnt += 1
        if cnt >= 5:
            break

    return stats2[0][0]

def mcts(board):
    root = { "plays": 0, "wins": 0 }

    iteration = 0
    best_move = None
    for iteration in range(0, 3000):
        select_move(board, root)

        iteration += 1
        if (iteration % 100) == 0:
            click.clear()  # Clear the screen
            print(board)
            print()
            print("Iteration {}".format(iteration))
            print()

            print("PV: {}".format(pv(board, root)))
            print()
            statistics(root)

    best_move = statistics(root)
    return best_move

if __name__ == "__main__":
    # board, _ = Board.from_epd("4r2k/p5pp/8/3Q1b1q/2B2P1P/P1P2n2/5PK1/R6R b - -")
    board = Board()
    opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5"
    # opening = "d4 d5"
    # opening = "e4 c5 Nf3 Nc6"
    for move in opening.split(" "):
        m = board.parse_san(move)
        board.push(m)

    while not board.is_game_over(claim_draw = True):
        print(board)
        best_move = mcts(board)
        print(best_move)
        board.push(board.parse_san(best_move))
