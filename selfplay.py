#!/usr/bin/env python3

import chess.pgn
import uuid

from chess import Board
from datetime import date
from pos_generator import generate_kxk, generate_kqk, generate_krk
from network import load_or_create_model
from mcts import MCTS
from prometheus_client import start_http_server, Counter

MAX_HALFMOVES_IN_GAME = 200

def format_root_moves(root, board):
    if root.visit_count == 0:
        return None

    root_moves = []
    for key, val in root.children.items():
        prop = val.visit_count / root.visit_count
        if prop >= 1e-3:
            root_moves.append("{}:{:.3f}".format(board.san(key), prop))

    return "q={:.3f}; p=[{}]".format(
        1.0 - root.value(),
        ", ".join(root_moves))


def compare_trees(t1, t2):
    assert t1.visit_count == t2.visit_count
    if t1.value() != t2.value():
        print("{} != {}".format(t1.value(), t2.value()))
        for move, node in t1.children.items():
            print("{}: {} {}    {} {}".format(move, node.value(), node.visit_count, t2.children[move].value(), t2.children[move].visit_count))
        raise AssertionError("Values not equal.")
    else:
        for move, node in t1.children.items():
            compare_trees(node, t2.children[move])

def selfplay(model, verbose=True, prefix=None):
    suffix = str(uuid.uuid4())
    mcts = MCTS(model, verbose, prefix, exploration_noise=True, max_simulations=800)

    total_positions = 0

    pos_counter = Counter('positions_total', "Positions generated by selfplay")
    game_counter = Counter('games_total', 'Games played by selfplay')

    while total_positions < 1638400:

        player = "Amy Zero [{}]".format(model.name)

        game = chess.pgn.Game()
        game.headers["Event"] = "Test Game"
        game.headers["White"] = player
        game.headers["Black"] = player
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        node = game

        board = Board()

        while not board.is_game_over(claim_draw = True) and board.halfmove_clock < MAX_HALFMOVES_IN_GAME:
            best_move, tree = mcts.mcts(board)

            # bm2, tree2 = mcts.mcts(board)
            # compare_trees(tree, tree2)

            node = node.add_variation(best_move)
            node.comment = format_root_moves(tree, board)

            board.push(best_move)

            if node.comment:
                total_positions += 1
                pos_counter.inc()

        game.headers["Result"] = board.result(claim_draw=True)

        with open("LearnGames-{}.pgn".format(suffix), "a") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

        game_counter.inc()

if __name__ == "__main__":

    start_http_server(9099)

    model = load_or_create_model("combined-model.h5")
    selfplay(model)
