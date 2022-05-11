import argparse
import pickle
import random
from functools import partial
from queue import PriorityQueue
from sys import exit
from threading import Thread

from pgn_reader import pos_generator

NFILES = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a training file.")
    parser.add_argument(
        "--split", type=int, help="Split percentage, default is 10%", default=10
    )
    parser.add_argument("filename")

    args = parser.parse_args()

    queue = PriorityQueue()

    pos_gen = partial(pos_generator, args.filename, True, queue)

    t = Thread(target=pos_gen)
    t.start()

    validation_file = open("data/validation.pkl", "wb")

    train_files = []
    for i in range(NFILES):
        train_files.append(open(f"data/train-{i}.pkl", "wb"))

    val_cnt = 0
    train_cnt = 0

    while train_cnt < 4_000_000:

        item = queue.get()
        if item.data_board is None:
            break

        is_validation = random.randint(0, 99) < args.split
        file = (
            validation_file
            if is_validation
            else train_files[random.randint(0, NFILES - 1)]
        )
        pickle.dump(item, file)

        if is_validation:
            val_cnt += 1
        else:
            train_cnt += 1

    validation_file.close()
    for f in train_files:
        f.close()

    print(f"Positions: {train_cnt}/{val_cnt} (training/validation)")

    exit(0)
