from math import sqrt, log10, inf
import chess.pgn
import argparse


def elo_confidence(win, loss, draw, k=1):
    N = win + loss + draw
    if N < 2:
        return 0, -inf, inf
    m = (win + 0.5*draw) / N
    stdev = sqrt((win*(1-m)**2 + loss*m**2 + draw*(0.5-m)**2) / (N - 1))

    xmin = m - k * stdev / sqrt(N)
    xmax = m + k * stdev / sqrt(N)

    if xmin > 0.0 and xmin < 1.0:
        min_elo = -400 * log10(1/xmin - 1)
    else:
        min_elo = -inf

    if xmax > 0.0 and xmax < 1.0:
        max_elo = -400 * log10(1/xmax - 1)
    else:
        max_elo = inf

    if m == 0.0:
        tp = -inf
    elif m == 1.0:
        tp = inf
    else:
        tp = -400 * log10(1/m - 1)

    return tp, min_elo, max_elo

def report_on_match(filename):
    stats = dict()

    with open(filename) as pgn:
        game_counter = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except UnicodeDecodeError or ValueError:
                continue

            if game is None: break

            game_counter += 1
            print("Parsing game #{}".format(game_counter), end='\r')

            result = game.headers["Result"]
            white = game.headers["White"]
            black = game.headers["Black"]

            stats_white = stats.get(white, None)
            if stats_white is None:
                stats_white = {'win': 0, 'loss': 0, 'draw': 0 }
                stats[white] = stats_white

            stats_black = stats.get(black, None)
            if stats_black is None:
                stats_black = {'win': 0, 'loss': 0, 'draw': 0 }
                stats[black] = stats_black

            if result == '1-0':
                stats_white['win'] += 1
                stats_black['loss'] += 1

            if result == '0-1':
                stats_white['loss'] += 1
                stats_black['win'] += 1

            if result == '1/2-1/2':
                stats_white['draw'] += 1
                stats_black['draw'] += 1

    for name, results in stats.items():
        n = results['win'] + results['loss'] + results['draw']
        if n > 1:
            elos = elo_confidence(results['win'], results['loss'], results['draw'])
            print("{:30} {:3}/{:3}/{:3}  {} [{}, {}] ".format(
                name,
                results['win'],
                results['loss'],
                results['draw'],
                round(elos[0], 0), round(elos[1], 0), round(elos[2], 0)
                ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ELO evaluation of a game file.")
    parser.add_argument("filename")

    args = parser.parse_args()

    report_on_match(args.filename)
