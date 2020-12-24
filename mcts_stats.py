from prometheus_client import Counter, Histogram
from colors import color
from pv import variations
from move_selection import select_root_move

import textwrap
import time
import numpy as np
import click

class MCTS_Stats:

    def __init__(self, model_name, verbose, prefix=None):
        self.max_depth = 0
        self.sum_depth = 0
        self.depth_list = []
        self.terminal_nodes = 0
        self.num_simulations = 0
        self.start_time = time.perf_counter()
        self.model_name = model_name
        self.verbose = verbose
        self.prefix = prefix
        self.best_move = None
        self.wrapper = textwrap.TextWrapper(
            initial_indent=9 * " ", subsequent_indent=11 * " ", width=119
        )

    def observe_depth(self, depth):
        self.max_depth = max(self.max_depth, depth)
        self.sum_depth += depth
        self.depth_list.append(depth)
        self.num_simulations += 1
        nodes_counter.inc()
        depth_histogram.observe(depth)

    def observe_terminal_node(self):
        self.terminal_nodes += 1
        terminal_nodes_counter.inc()

    def statistics(self, root, board):

        elapsed = time.perf_counter() - self.start_time

        if self.verbose:
            avg_depth = self.sum_depth / self.num_simulations

            click.clear()

            print(
                f"{board}   {'White' if board.turn else 'Black'}: {self.model_name}"
            )
            print()
            print(board.fen())
            # print()
            # print(board.variation_san(principal_variation))
            print()
            print(
                "{} simulations in {:.1f} seconds = {:.1f} simulations/sec".format(
                    self.num_simulations, elapsed, self.num_simulations / elapsed
                )
            )
            print()
            print(
                "Max depth: {}  Avg depth: {:.1f}  Terminal nodes: {:.1f}%".format(
                    self.max_depth,
                    avg_depth,
                    100 * self.terminal_nodes / self.num_simulations,
                )
            )
            print()

            stats = [
                (key, val) for key, val in root.children.items() if val.visit_count > 0
            ]
            stats = sorted(stats, key=lambda e: e[1].visit_count, reverse=True)

            variations_cnt = 3
            print(" Score Line      Visit-% [prior]")
            print()

            if len(stats) > 0:
                current_best_move = stats[0][0]
                if self.best_move is None or self.best_move != current_best_move:
                    self.best_move = current_best_move
                    self.best_move_found = self.num_simulations

            message = ""
            if self.best_move is not None:
                message = "   (since iteration {})".format(self.best_move_found)

            for move, child_node in stats[:10]:
                print(
                    color(
                        "{:5.1f}% {:10s} {:5.1f}% [{:4.1f}%] {:6d} visits {}".format(
                            100 * child_node.value(),
                            board.variation_san([move]),
                            100 * child_node.visit_count / self.num_simulations,
                            100 * child_node.prior,
                            child_node.visit_count,
                            message,
                        ),
                        get_color(child_node.value()),
                    )
                )
                message = ""
                if variations_cnt > 0:
                    variations_list = variations(
                        board, move, child_node, variations_cnt
                    )
                    for variation in variations_list:
                        for line in self.wrapper.wrap(variation):
                            print(line)
                    print()
                    variations_cnt -= 1

            if len(stats) > 10:
                print()
                remaining_moves = []
                for move, child_node in stats[10:]:
                    remaining_moves.append(
                        color(
                            "{} ({:.1f}%, {})".format(
                                board.san(move),
                                100 * child_node.value(),
                                child_node.visit_count,
                            ),
                            get_color(child_node.value()),
                        )
                    )
                print(", ".join(remaining_moves))

        else:
            best_move = select_root_move(root, board.fullmove_number, False)
            variations_list = variations(board, best_move, root.children[best_move], 1)
            if len(variations_list) == 0:
                variations_list.append("")

            print(
                color(
                    "{:3} - {:4} {:5.1f}% {:12} [{:.1f} sims/s]  {}".format(
                        self.prefix or "",
                        self.num_simulations,
                        100 * root.children[best_move].value(),
                        board.variation_san([best_move]),
                        self.num_simulations / elapsed,
                        variations_list[0],
                    ),
                    get_color(root.children[best_move].value()),
                )
            )

nodes_counter = Counter("nodes", "Nodes visited")
terminal_nodes_counter = Counter("terminal_nodes", "Terminal nodes visited")
depth_histogram = Histogram(
    "depth",
    "Search depth",
    buckets=[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20,
        24,
        28,
        32,
        36,
        40,
        48,
        56,
        64,
    ],
)


def get_color(x):
    t = min(int(x * 13), 12)
    if t < 6:
        return 16 + 5 * 36 + 6 * t + t
    elif t > 6:
        t = 12 - t
        return 16 + 36 * t + 6 * 5 + t
    else:
        return 0
