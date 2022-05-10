import glob
import os
import random

SAMPLE_RATE=10
POSITION_COUNT=600_000

def find_train_files(position_count: int, sample_rate: int, test_mode: bool):
    pfiles = glob.glob('data/*/position_count')

    total_count = 0

    data_files = []

    for name in sorted(pfiles, reverse=True):
        with open(name) as pfile:
            cnt = int(pfile.readline())
        total_count += cnt
        print(f"{name} {cnt} {total_count}")
        dir = os.path.dirname(name)
        print(dir)

        if test_mode:
           data_files.extend(glob.glob(os.path.join(dir, "validation.pkl")))
        else:
           data_files.extend(glob.glob(os.path.join(dir, "train-*.pkl")))

        if total_count * sample_rate / 100 > position_count:
            break

    random.shuffle(data_files)
    return data_files


if __name__ == "__main__":

    find_train_files(POSITION_COUNT, SAMPLE_RATE, False)
