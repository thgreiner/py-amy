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

from mcts import mcts

if __name__ == "__main__":

    name = sys.argv[1]
    
    with open(name, "r") as f:
        
        while True:
            l = f.readline()
        
            b = Board()
            b.set_epd(l)
        
            mcts(b)