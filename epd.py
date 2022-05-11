#!/usr/bin/env python3

import argparse
import math
import random
import sys
import time
import uuid
from datetime import date

import chess.pgn
import click
import numpy as np
import tensorflow_model_optimization as tfmot
from chess import Board
from prometheus_client import Counter, start_http_server

import piece_square_eval
from chess_input import Repr2D
from move_selection import select_root_move
from network import load_or_create_model
from pos_generator import generate_kxk
from searcher import AmySearcher, Searcher

if __name__ == "__main__":

    solved = 0
    total = 0

    parser = argparse.ArgumentParser(description="Run evaluation on a EPD file.")
    parser.add_argument("--model", help="model file name")
    parser.add_argument("filename")
    parser.add_argument(
        "--sims", type=int, help="number of simulations", default=2_000_000
    )

    args = parser.parse_args()

    start_http_server(9100)

    if args.model == "tflite":
        from edgetpu import EdgeTpuModel
        from mcts import MCTS

        model = EdgeTpuModel("models/tflite-104x15_edgetpu.tflite")
    elif args.model.endswith("_edgetpu.tflite"):
        from edgetpu import EdgeTpuModel
        from mcts import MCTS

        model = EdgeTpuModel(args.model)
    elif args.model == "tensorrt":
        from mcts_batched import MCTS
        from tensorrt_model import TensorRTModel

        model = TensorRTModel()
    else:
        from mcts_batched import MCTS

        with tfmot.quantization.keras.quantize_scope():
            model = load_or_create_model(args.model)

    mcts = MCTS(
        model,
        prefix=None,
        max_simulations=args.sims,
        exploration_noise=False,
        verbose=True,
    )

    with open(args.filename, "r") as f:

        for l in f:
            b, tags = Board.from_epd(l)

            tree = mcts.mcts(b, prefix="")

            if isinstance(tags, dict):
                best_moves = tags.get("bm", [])
                move_found = select_root_move(tree, 0, False)

                total += 1

                if move_found in best_moves:
                    solved += 1
                    print(f"Solved! {solved}/{total}")
                else:
                    print(f"Not solved! {solved}/{total}")
