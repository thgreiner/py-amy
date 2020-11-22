#!/usr/bin/env python3
from math import log
from random import uniform
import numpy as np


def sample_gumbel(a):
    b = [log(x) - log(-log(uniform(0, 1))) for x in a]
    return np.argmax(b)


def select_root_move(tree, move_count, sample=True):

    if len(tree.children) == 0:
        return None

    k = 2.0
    moves = []
    visits = []
    for key, val in tree.children.items():
        if val.visit_count > 0:
            moves.append(key)
            visits.append(val.visit_count ** k)

    if sample and move_count < 15:
        idx = sample_gumbel(visits)
    else:
        idx = np.argmax(visits)

    return moves[idx]


def select_root_move_delta(tree, move_count, sample=True, delta=0.02):

    if len(tree.children) == 0:
        return None

    _, best_value = max(
        ((child.visit_count, child.value()) for action, child in tree.children.items()),
        key=lambda e: e[0],
    )

    k = 2.0
    moves = []
    visits = []
    for key, val in tree.children.items():
        if val.visit_count > 0 and val.value() > (best_value - delta):
            moves.append(key)
            visits.append(val.visit_count ** k)
            # print("{} {:4.1f}% {:4d}".format(key, 100*val.value(), val.visit_count))

    if sample and move_count < 15:
        idx = sample_gumbel(visits)
    else:
        idx = np.argmax(visits)

    return moves[idx]


def add_exploration_noise(node):
    root_dirichlet_alpha = 0.3
    root_exploration_fraction = 0.25

    actions = node.children.keys()
    noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
    noise /= np.sum(noise)
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
