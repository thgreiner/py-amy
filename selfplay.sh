#!/bin/bash

while true; do
    if [ -f LearnGames.pgn ]; then
        DATE=$(date "+%Y-%m-%d-%H-%M")
        mv LearnGames.pgn LearnGames-$DATE.pgn
    fi

    python3 mcts.py
    python3 train_from_pgn.py LearnGames.pgn combined-model.h5
done
