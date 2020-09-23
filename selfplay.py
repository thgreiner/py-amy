#!/usr/bin/env python3

import chess.pgn
import uuid
import argparse

from chess import Board
from datetime import date
import pos_generator
from network import load_or_create_model
from mcts import MCTS, select_root_move
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

def create_node_with_comment(node, tree, best_move, board, pos_counter ,follow_main_line=False):
    new_node = node.add_main_variation(best_move)
    new_node.comment = format_root_moves(tree, board)

    if new_node.comment:
        pos_counter.inc()

    for move, subtree in tree.children.items():
        if not follow_main_line and move == best_move: continue

        if subtree.visit_count > 100:  # arbitrary threshold
            if follow_main_line and move == best_move:
                tmp_node = new_node
            else:
                tmp_node = node.add_variation(move)
            board.push(move)
            tmp_best_move = select_root_move(subtree, board.fullmove_number, False)
            if tmp_best_move is not None:
                create_node_with_comment(tmp_node, subtree, tmp_best_move, board, pos_counter, True)
            board.pop()

    return new_node

# not really nice here...
start_http_server(9099)
pos_counter = Counter('positions_total', "Positions generated by selfplay")
game_counter = Counter('games_total', 'Games played by selfplay')

def selfplay(model, num_simulations, verbose=True, prefix="0", generator=None):
    suffix = str(uuid.uuid4())
    mcts = MCTS(model, verbose, prefix, exploration_noise=True, max_simulations=num_simulations)

    total_positions = 0

    while total_positions < 1638400:

        player = "Amy Zero [{}]".format(model.name)

        game = chess.pgn.Game()
        game.headers["Event"] = "Test Game"
        game.headers["White"] = player
        game.headers["Black"] = player
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        node = game

        if generator:
            board = generator()
            game.setup(board)
        else:
            board = Board()

        while not board.is_game_over(claim_draw = True) and board.halfmove_clock < MAX_HALFMOVES_IN_GAME:
            best_move, tree = mcts.mcts(board, prefix=prefix)

            node = create_node_with_comment(node, tree, best_move, board, pos_counter)

            board.push(best_move)

            if node.comment:
                total_positions += 1

        game.headers["Result"] = board.result(claim_draw=True)

        with open("LearnGames-{}.pgn".format(suffix), "a") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

        game_counter.inc()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Self play.")
    parser.add_argument('--sims', type=int, help="number of simulations", default=800)
    parser.add_argument('--kqk', action='store_const', const=True, default=False, help="only play K+Q vs K games")
    parser.add_argument('--krk', action='store_const', const=True, default=False, help="only play K+R vs K games")
    parser.add_argument('--kxk', action='store_const', const=True, default=False, help="only play K+X vs K games")
    parser.add_argument('--kpkp', action='store_const', const=True, default=False, help="only play K+P vs K+P games")
    parser.add_argument('--kqqk', action='store_const', const=True, default=False, help="only play K+Q+Q vs K games")

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
    if args.kqqk:
        generator = pos_generator.generate_kqqk

    model = load_or_create_model("combined-model.h5")
    selfplay(model, args.sims, generator=generator)
