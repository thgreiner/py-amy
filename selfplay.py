#!/usr/bin/env python3

import chess.pgn
import argparse
import random
import time

from chess import Board
import pos_generator
from network import load_or_create_model
from move_selection import select_root_move
from pgn_writer import DefaultGameSaver, create_node_with_comment
from prometheus_client import start_http_server

MAX_HALFMOVES_IN_GAME = 200


def selfplay(
    model,
    num_simulations,
    verbose=True,
    prefix="0",
    generator=None,
    saver=None,
    eco=False,
):

    mcts = MCTS(
        model, verbose, prefix, exploration_noise=True, max_simulations=num_simulations
    )
    mcts.set_kldgain_stop(2.0e-3)

    if saver is None:
        saver = DefaultGameSaver("LearnGames")

    total_positions = 0
    round = 0

    if eco:
        eco_in = open("PGN/eco.pgn", "r", encoding="ISO-8859-1")
        print("Playing through ECO positions.")

    while total_positions < 1638400:

        player = "Amy Zero [{}]".format(model.name)

        round += 1

        game = chess.pgn.Game()
        game.headers["Event"] = "Test Game"
        game.headers["White"] = player
        game.headers["Black"] = player
        game.headers["Round"] = str(round)
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        node = game

        if generator:
            board = generator()
            game.setup(board)
        else:
            board = Board()

        fully_playout_game = random.randint(0, 100) < 25

        if eco:
            fully_playout_game = True
            while True:
                line = eco_in.readline().split()
                if len(line) > 12:
                    break
            line.reverse()
        else:
            line = None

        while (
            not board.is_game_over(claim_draw=True)
            and board.halfmove_clock < MAX_HALFMOVES_IN_GAME
        ):
            is_full_playout = fully_playout_game or (random.randint(0, 100) < 25)

            try:
                bias_move = board.parse_san(line.pop()) if line else None
            except ValueError:
                bias_move = None
                line = None

            if bias_move or is_full_playout:
                tree = mcts.mcts(board, prefix=prefix, bias_move=bias_move)
                best_move = (
                    bias_move
                    if bias_move
                    else select_root_move(tree, board.fullmove_number, True)
                )
                node = create_node_with_comment(
                    node, tree, best_move, board
                )
            else:
                tree = mcts.mcts(board, prefix=prefix, limit=100)
                best_move = select_root_move(tree, board.fullmove_number, True)
                node = node.add_main_variation(best_move)

            board.push(best_move)

            if node.comment:
                total_positions += 1

        game.headers["Result"] = board.result(claim_draw=True)
        saver(game)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Self play.")
    parser.add_argument("--sims", type=int, help="number of simulations", default=800)
    parser.add_argument("--model", help="model file name")
    parser.add_argument(
        "--eco",
        action="store_const",
        const=True,
        default=False,
        help="play through standard ECO openings",
    )
    parser.add_argument(
        "--kqk",
        action="store_const",
        const=True,
        default=False,
        help="only play K+Q vs K games",
    )
    parser.add_argument(
        "--kqkr",
        action="store_const",
        const=True,
        default=False,
        help="only play K+Q vs K+R games",
    )
    parser.add_argument(
        "--krk",
        action="store_const",
        const=True,
        default=False,
        help="only play K+R vs K games",
    )
    parser.add_argument(
        "--kxk",
        action="store_const",
        const=True,
        default=False,
        help="only play K+X vs K games",
    )
    parser.add_argument(
        "--kpkp",
        action="store_const",
        const=True,
        default=False,
        help="only play K+P vs K+P games",
    )
    parser.add_argument(
        "--kqqk",
        action="store_const",
        const=True,
        default=False,
        help="only play K+Q+Q vs K games",
    )

    args = parser.parse_args()

    generator = None
    if args.kqk:
        generator = pos_generator.generate_kqk
    if args.krk:
        generator = pos_generator.generate_krk
    if args.kxk:
        generator = pos_generator.generate_kxk
    if args.kpkp:
        generator = pos_generator.generate_kpkp
    if args.kqkr:
        generator = pos_generator.generate_kqkr
    if args.kqqk:
        generator = pos_generator.generate_kqqk

    if args.model == "tflite":
        from mcts import MCTS
        from edgetpu import EdgeTpuModel

        model = EdgeTpuModel("models/tflite-128x19_edgetpu.tflite")
    elif args.model.endswith("_edgetpu.tflite"):
        from mcts import MCTS
        from edgetpu import EdgeTpuModel

        model = EdgeTpuModel(args.model)
    else:
        from mcts_batched import MCTS

        model = load_or_create_model(args.model)

    for port in range(9100, 9104):
        try:
            start_http_server(port)
            print(f"Started http server on port {port}")
            break
        except OSerror:
            pass

    selfplay(model, args.sims, generator=generator, verbose=True, eco=args.eco)
