#!/usr/bin/env python3

import chess.pgn
import argparse
import random
import time

from chess import Board
from datetime import date
import pos_generator
from network import load_or_create_model
from mcts import MCTS, select_root_move
from prometheus_client import start_http_server, Counter
from pgn_writer import DefaultGameSaver, create_node_with_comment
from move_selection import select_root_move_delta

MAX_HALFMOVES_IN_GAME = 200


def matchplay(
    model1, model2, num_simulations, verbose=True, prefix="0", generator=None
):

    mcts1 = MCTS(
        model1,
        verbose,
        prefix,
        exploration_noise=False,
        max_simulations=num_simulations,
    )
    mcts2 = MCTS(
        model2,
        verbose,
        prefix,
        exploration_noise=False,
        max_simulations=num_simulations,
    )

    saver = DefaultGameSaver("MatchGames")

    total_positions = 0
    round = 0

    while total_positions < 1638400:

        player1 = "Amy Zero [{}]".format(mcts1.model_name())
        player2 = "Amy Zero [{}]".format(mcts2.model_name())

        round += 1

        game = chess.pgn.Game()
        game.headers["Event"] = "Test Game"
        game.headers["White"] = player1
        game.headers["Black"] = player2
        game.headers["Round"] = str(round)
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        node = game

        board = Board()

        while (
            not board.is_game_over(claim_draw=True)
            and board.halfmove_clock < MAX_HALFMOVES_IN_GAME
        ):
            if board.turn:
                tree = mcts1.mcts(board, prefix=prefix)
            else:
                tree = mcts2.mcts(board, prefix=prefix)

            best_move = select_root_move_delta(tree, board.fullmove_number)

            node = create_node_with_comment(node, tree, best_move, board)

            board.push(best_move)

            if node.comment:
                total_positions += 1

        game.headers["Result"] = board.result(claim_draw=True)
        saver(game)

        mcts1, mcts2 = mcts2, mcts1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Self play.")
    parser.add_argument("--sims", type=int, help="number of simulations", default=5000)
    parser.add_argument("--model1", help="model1 file name")
    parser.add_argument("--model2", help="model2 file name")

    args = parser.parse_args()

    model1 = load_or_create_model(args.model1)
    model2 = load_or_create_model(args.model2)

    start_http_server(9100)

    matchplay(model1, model2, args.sims)
