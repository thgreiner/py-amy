from chess import Board
import random
from math import sqrt
from math import log

from chess_input import Repr2D

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model("move-model.h5")
repr = Repr2D()

C = 1.2


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


def playout(board, depth = 100):
    if board.is_game_over(claim_draw = True) or depth <= 0:
        return board.result(claim_draw = True)

    m = choose_move(board, model)
    # print("{} ".format(board.san(m)), end='')

    board.push(m)
    winner = playout(board, depth-1)
    board.pop()
    return winner


def score(board, winner):
    if board.turn and winner == "1-0":
        return 1
    if not board.turn and winner == "0-1":
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0

def select_move(board, node):
    if board.is_game_over(claim_draw = True):
        winner = board.result(claim_draw = True)
        return winner

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
        print("{} [".format(board.san(m)), end='')
        board.push(m)
        winner = playout(board)
        board.pop()

        print("] {}".format(winner))

        d = { "plays": 1, "wins": score(board, winner)}
        node[board.san(m)] = d
        node["plays"] += 1
        # node["wins"] += score(board, winner)

        return winner

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

        print("{} ".format(board.san(selected_move)), end='')

        board.push(selected_move)
        winner = select_move(board, selected_child_node)
        board.pop()

        # print("winner: {}".format(winner))

        node["plays"] += 1
        selected_child_node["wins"] += score(board, winner)

        return winner


def statistics(node):
    best_move = None
    best_visits = None
    best_wins = None

    stats = []
    for key, val in node.items():
        if isinstance(val, dict):
            win_ratio = val["wins"] / val["plays"]
            stats.append((key, win_ratio, val["plays"]))

    stats = sorted(stats, key = lambda e: e[1], reverse=True)
    cnt = 0
    for stat in stats:
        print("{:5s} {:.1f}% {} visits".format(stat[0], 100 * stat[1], stat[2]))
        cnt += 1
        if cnt >= 5:
            break

def mcts(board):
    root = { "plays": 0, "wins": 0 }

    iterations = 0
    while True:
        iterations += 1
        print("Iteration {}".format(iterations))
        select_move(board, root)
        statistics(root)

if __name__ == "__main__":
    board, _ = Board.from_epd("4r2k/p5pp/8/3Q1b1q/2B2P1P/P1P2n2/5PK1/R6R b - -")
    print(board)
    mcts(board)
