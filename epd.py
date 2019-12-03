#!/usr/bin/env python3

from chess import Board
import chess.pgn
import random
import math
import numpy as np
import time
import uuid
import sys

from datetime import date

from chess_input import Repr2D

import click

from searcher import Searcher, AmySearcher
import piece_square_eval
from pos_generator import generate_kxk

from network import load_or_create_model

from mcts import MCTS

from prometheus_client import start_http_server, Counter

if __name__ == "__main__":

    start_http_server(9099)

    name = sys.argv[1]

    model = load_or_create_model("combined-model.h5")
    mcts = MCTS(model, True, None, max_simulations = 5000000, exploration_noise=False)


    with open(name, "r") as f:

        while True:
            l = f.readline()

            b = Board()
            b.set_epd(l)

            mcts.mcts(b)
