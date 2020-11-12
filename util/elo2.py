#!/usr/bin/env python3

from math import sqrt, log10, inf
import chess.pgn
import argparse

DEFAULT_ELO=2000
K=20

def report_on_match(filename):
    stats = dict()

    for iters in range(100):
        print(f"Iteration {iters}")

        wins, losses, draws = dict(), dict(), dict()
        elo_delta = dict()

        with open(filename) as pgn:
            game_counter = 0
            while True:
                try:
                    headers = chess.pgn.read_headers(pgn)
                except UnicodeDecodeError or ValueError:
                    continue

                if headers is None: break

                game_counter += 1
                # print(f"Parsing game #{game_counter}", end='\r')

                result = headers["Result"]
                white = headers["White"]
                black = headers["Black"]

                elo_white = stats.get(white, DEFAULT_ELO)
                elo_black = stats.get(black, DEFAULT_ELO)

                ea = 1.0 / (1.0 + 10.0 ** ((elo_black-elo_white) / 400))
                eb = 1.0 - ea

                if result == '1-0':
                    wins[white] = wins.get(white, 0) + 1
                    losses[black] = losses.get(black, 0) + 1
                    delta_white = K * (1.0 - ea)
                    delta_black = K * (-eb)

                if result == '0-1':
                    losses[white] = losses.get(white, 0) + 1
                    wins[black] = wins.get(black, 0) + 1
                    delta_white = K * (-ea)
                    delta_black = K * (1.0 - eb)

                if result == '1/2-1/2':
                    draws[white] = draws.get(white, 0) + 1
                    draws[black] = draws.get(black, 0) + 1
                    delta_white = K * (0.5 - ea)
                    delta_black = K * (0.5 - eb)

                elo_delta[white] = elo_delta.get(white, 0) + delta_white
                elo_delta[black] = elo_delta.get(black, 0) + delta_black

        for name, delta in elo_delta.items():
            stats[name] = stats.get(name, DEFAULT_ELO) + delta * .5

        elo_ratings_by_name = sorted([(name, result) for name, result in stats.items()], key=lambda x: x[1])
        last_elo = None
        for name, elo in elo_ratings_by_name:
            if last_elo is not None:
                diff =  f"+{int(elo - last_elo)}"
            else:
                diff = ""
            print(f"{name:45} : {int(elo)} {diff:4}  +{wins.get(name,0)} -{losses.get(name,0)} ={draws.get(name, 0)}")
            last_elo = elo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ELO evaluation of a game file.")
    parser.add_argument("filename")

    args = parser.parse_args()

    report_on_match(args.filename)
