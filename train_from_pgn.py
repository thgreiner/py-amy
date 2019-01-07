import chess
from chess import Board
import numpy as np
import chess.pgn
from chess_input import Repr2D

import sys
import time
import random
import argparse

from network import load_or_create_model

# Training batch size
BATCH_SIZE = 256

# Checkpoint every x batches
CHECKPOINT = 200


p1 = 0.5
p2 = 0.3

c = (p1 / p2) ** (1/20)

def drop_move(fullmove_number):
    prob_of_dropping = p1 * (c ** fullmove_number)
    return random.uniform(0, 1) < prob_of_dropping


def label_for_result(result, turn):
    if result == '1-0':
        if turn:
            return 1
        else:
            return -1
    if result == '0-1':
        if turn:
            return -1
        else:
            return 1

    return 0


def stats(step_output):
    loss = step_output[0]
    moves_loss = step_output[1]
    score_loss = step_output[2]
    reg_loss = loss - moves_loss - score_loss

    moves_accuracy = step_output[3]
    score_mae = step_output[6]

    return "loss: {:.2f} = {:.2f} + {:.2f} + {:.2f}, move accuracy: {:2.0f}%, score mae: {:.2f}".format(
        loss,
        moves_loss, score_loss, reg_loss,
        moves_accuracy * 100, score_mae
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training on a PGN file.")
    parser.add_argument("filename")
    parser.add_argument('--diff', type=int, help="minimum elo barrier", default=0)
    parser.add_argument('--model', help="model file name")

    args = parser.parse_args()

    repr = Repr2D()

    elo_diff = args.diff
    model_name = args.model
    model = load_or_create_model(model_name)

    train_data_board = np.zeros(((BATCH_SIZE, 8, 8, repr.num_planes)), np.int8)
    train_data_moves = np.zeros((BATCH_SIZE, 4672), np.int8)
    train_labels1 = np.zeros((BATCH_SIZE, 4672), np.int8)
    train_labels2 = np.zeros((BATCH_SIZE, 1), np.float32)

    start_time = time.perf_counter()

    for iteration in range(100):
        with open(args.filename) as pgn:
            cnt = 0

            ngames = 0

            samples = 0
            checkpoint_no = 0
            checkpoint_next = CHECKPOINT * BATCH_SIZE

            while True:
                # skip = random.uniform(0, 100)
                # for i in range(int(skip)):
                #     if not chess.pgn.skip_game(pgn):
                #        break

                try:
                    game = chess.pgn.read_game(pgn)
                except UnicodeDecodeError or ValueError:
                    pass
                if game is None:
                    break

                # if label == 0:
                #     continue
                result = game.headers["Result"]
                white = game.headers["White"]
                black = game.headers["Black"]
                date_of_game = game.headers["Date"]

                if "WhiteElo" in game.headers:
                    white_elo = game.headers["WhiteElo"]
                else:
                    white_elo = "-"

                if "BlackElo" in game.headers:
                    black_elo = game.headers["BlackElo"]
                else:
                    black_elo = "-"

                if white_elo != "-" and black_elo != "-":
                    w = int(white_elo)
                    b = int(black_elo)
                    if abs(w - b) < elo_diff:
                        # print("Skipping game - Elo diff less than {}}.".format(elo_diff))
                        continue
                else:
                    # print("Skipping game, one side has no Elo.")
                    continue

                print("{}: {} ({}) - {} ({}), {} {}". format(
                    ngames,
                    white, white_elo,
                    black, black_elo,
                    result, date_of_game))

                ngames += 1

                b = game.board()
                nmoves = 0
                moves_in_game = len(list(game.mainline_moves()))

                try:
                    for move in game.mainline_moves():

                        if not drop_move(b.fullmove_number):
                            train_data_board[cnt] = repr.board_to_array(b)
                            train_data_moves[cnt] = repr.legal_moves_mask(b)
                            train_labels1[cnt] = repr.move_to_array(b, move)
                            train_labels2[cnt, 0] = label_for_result(result, b.turn)
                            cnt += 1

                            if cnt >= BATCH_SIZE:
                                # print(train_labels2)
                                train_data = [ train_data_board, train_data_moves]
                                train_labels = [ train_labels1, train_labels2 ]

                                results = model.train_on_batch(train_data, train_labels)
                                elapsed = time.perf_counter() - start_time

                                samples += cnt
                                print("{}.{}: {} in {:.1f}s [{} games]".format(
                                    iteration, samples, stats(results), elapsed, ngames))
                                start_time = time.perf_counter()

                                cnt = 0
                                if samples >= checkpoint_next:
                                    checkpoint_no += 1
                                    checkpoint_name = "checkpoint-{}.h5".format(checkpoint_no)
                                    print("Checkpointing model to {}".format(checkpoint_name))
                                    model.save(checkpoint_name)
                                    checkpoint_next += CHECKPOINT * BATCH_SIZE

                        b.push(move)
                        nmoves += 1
                except AttributeError:
                    print("Oops - bad game encountered. Skipping it...")

            # Train on the remainder of the dataset
            train_data = [ train_data_board[:cnt], train_data_moves[:cnt]]
            train_labels = [ train_labels1[:cnt], train_labels2[:cnt] ]

            start_time = time.perf_counter()
            results = model.train_on_batch(train_data, train_labels)
            elapsed = time.perf_counter() - start_time

            samples += cnt
            print("{}.{}: {} in {:.1f}s [{} games]".format(
                iteration, samples, stats(results), elapsed, ngames))

            if model_name is None:
                model.save("combined-model.h5")
            else:
                model.save(model_name)
