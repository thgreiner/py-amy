#!/usr/bin/env python3

import chess.pgn
import uuid

from chess import Board
from datetime import date
from pos_generator import generate_kxk
from network import load_or_create_model
from mcts import MCTS

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
        1.0 - root.value_sum / root.visit_count,
        ", ".join(root_moves))


def selfplay(model, verbose=True, prefix=None):
    suffix = str(uuid.uuid4())
    mcts = MCTS(model, verbose, prefix)

    total_positions = 0
    while total_positions < 16384:

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
            node = node.add_variation(best_move)
            node.comment = format_root_moves(tree, board)

            board.push(best_move)
            total_positions += 1

        game.headers["Result"] = board.result(claim_draw=True)

        with open("LearnGames-{}.pgn".format(suffix), "a") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)


if __name__ == "__main__":

    model = load_or_create_model("combined-model.h5")
    selfplay(model)
