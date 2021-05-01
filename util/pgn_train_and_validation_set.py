#!/usr/bin/env python3

import chess.pgn
import os
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an PGN file into test and validation set.")
    parser.add_argument("filename")
    parser.add_argument('--split', type=int, help="Split percentage, default is 10%", default=10)

    args = parser.parse_args()

    with open(args.filename) as pgn, \
         open("train.pgn", "w") as train_file, \
         open("validation.pgn", "w") as validation_file:

        cnt=0

        train_writer = chess.pgn.FileExporter(train_file, variations=False)
        validation_writer = chess.pgn.FileExporter(validation_file, variations=False)

        while True:

            cnt += 1

            game = chess.pgn.read_game(pgn)
            if game is None: break

            date_of_game = game.headers["Date"]
            print("Parsing game #{}: {}".format(cnt, date_of_game), end='\r')

            is_validation = random.randint(0, 100) < args.split

            writer = validation_writer if is_validation else train_writer

            game.accept(writer)
