#!/usr/bin/env python3

import chess.pgn
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an PGN file by date.")
    parser.add_argument("filename")

    args = parser.parse_args()

    with open(args.filename) as pgn:
    
        while True:
        
            game = chess.pgn.read_game(pgn)
            if game is None: break

            date = game.headers["Date"]
            output_dir= "PGN/{}".format(date)
        
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        
            with open("{}/games.pgn".format(output_dir), "a") as output:
                print(game, file=output, end="\n\n")
            
            print(date)