#!/usr/bin/env python3

from chess import Board
import chess.pgn
import random
import math
import numpy as np
import time
import uuid
import sys
import argparse

from datetime import date

from chess_input import Repr2D

import click

from searcher import Searcher, AmySearcher
import piece_square_eval
from pos_generator import generate_kxk
from move_selection import select_root_move_delta
from pgn_writer import DefaultGameSaver, create_node_with_comment

from network import load_or_create_model

from mcts import MCTS

from prometheus_client import start_http_server

MAX_HALFMOVES_IN_GAME = 200

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evaluation on a EPD file.")
    parser.add_argument('--model', help="model file name")

    args = parser.parse_args()

    start_http_server(9100)

    model = load_or_create_model(args.model)

    mcts = MCTS(model, max_simulations=800)

    ps_searcher = Searcher(lambda board: piece_square_eval.evaluate(board), "PieceSquareTables")

    saver=DefaultGameSaver("TestGames")

    total_positions = 0
    while total_positions < 16384:

        game = chess.pgn.Game()
        game.headers["Event"] = "Test Game"
        game.headers["White"] = "Amy Zero"
        game.headers["Black"] = "Amy Zero"
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")

        tree = None
        # board, _ = Board.from_epd("4r2k/p5pp/8/3Q1b1q/2B2P1P/P1P2n2/5PK1/R6R b - -")

        board = Board()
        # board = generate_kxk()
        # board.set_fen("8/k7/5Q2/8/8/8/8/4K3 b - - 0 1")

        opening = None
        # opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5"
        # opening = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 Nbd7 Nf3 O-O Bd3 dxc4 Bxc4 c6 O-O b5 Bd3 h6 Bf4 b4 Ne4 Nxe4 Bxe4 Ba6 Qa4 Bb5"
        opening = "d4 d5"
        # opening = "d4 d5 c4 e6 Nc3 Nf6"
        # opening = "e4 c5 Nf3 Nc6"

        node = game
        if opening:
            for move in opening.split(" "):
                m = board.parse_san(move)
                board.push(m)
                node = node.add_variation(m)

        while not board.is_game_over(claim_draw = True) and board.halfmove_clock < MAX_HALFMOVES_IN_GAME:
            if board.turn:
                tree = mcts.mcts(board, prefix=None)
                best_move = select_root_move_delta(tree, board.fullmove_number)
                node = create_node_with_comment(node, tree, best_move, board)
            else:
                best_move = ps_searcher.select_move(board)
                node = node.add_variation(best_move)

            board.push(best_move)
            total_positions += 1

        game.headers["Result"] = board.result(claim_draw=True)

        saver(game)
