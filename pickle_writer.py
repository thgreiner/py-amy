from pgn_reader import pos_generator

from threading import Thread
from queue import PriorityQueue
import random

import argparse
import pickle
from functools import partial

NFILES=10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a training file.")
    parser.add_argument('--split', type=int, help="Split percentage, default is 10%", default=10)
    parser.add_argument("filename")

    args = parser.parse_args()

    queue = PriorityQueue()

    pos_gen = partial(pos_generator, args.filename, False, queue)

    t = Thread(target=pos_gen)
    t.start()

    validation_file = open("data/validation.pkl", "wb")

    train_files = []
    for i in range(NFILES):
        train_files.append(open(f"data/train-{i}.pkl", "wb"))

    while True:

        item = queue.get()
        if item.data_board is None:
            break

        is_validation = random.randint(0, 99) < args.split
        file = validation_file if is_validation else train_files[random.randint(0, NFILES-1)]
        pickle.dump(item, file)

    validation_file.close()
    for f in train_files: f.close()
