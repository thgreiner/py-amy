from chess import Board
from random import choice
from math import sqrt
from math import log

C = 1.414

def playout(board, depth = 100):
    if board.is_game_over(claim_draw = True) or depth <= 0:
        return board.result(claim_draw = True)
    moves = list(board.generate_pseudo_legal_moves())
    while moves:
        m = choice(moves)
        if board.is_legal(m):
            board.push(m)
            winner = playout(board, depth-1)
            board.pop()
            return winner
    return None


def score(board, winner):
    if board.turn and winner == "1-0":
        return 1
    if not board.turn and winner == "0-1":
        return 1
    if winner == "1/2-1/2" or winner == "*":
        return 0.5
    return 0

def select_move(board, node):
    moves = list(board.generate_legal_moves())

    visited = list()
    non_visited = list()

    for m in moves:
        san = board.san(m)
        if san in node:
            visited.append(m)
        else:
            non_visited.append(m)

    # print("visited: {}".format(visited))
    # print("non_visited: {}".format(non_visited))

    if non_visited:
        m = choice(non_visited)
        board.push(m)
        winner = playout(board)
        d = { "plays": 1, "wins": score(board, winner)}
        board.pop()

        print("winner: {}".format(winner))

        node[board.san(m)] = d
        node["plays"] += 1
        node["wins"] += score(board, winner)

        return winner

    else:
        visit_count = node["plays"]
        print("Visit count: {}".format(visit_count))

        selected_move = None
        selected_prob = None
        selected_child_node = None

        for m in moves:
            san = board.san(m)
            child = node[san]

            child_wins = child["wins"]
            child_plays = child["plays"]

            child_prob = child_wins / child_plays + C * sqrt(log(visit_count) / child_plays)
            # print("{}: plays:{} wins:{} {}".format(san, child_plays, child_wins, child_prob))

            if selected_move is None or child_prob > selected_prob:
                selected_move = m
                selected_prob = child_prob
                selected_child_node = child

        print("Selected move: {}".format(board.san(selected_move)))

        board.push(selected_move)
        winner = select_move(board, selected_child_node)
        board.pop()

        print("winner: {}".format(winner))

        node["plays"] += 1
        node["wins"] += score(board, winner)

        return winner


def statistics(node):
    best_move = None
    best_visits = None
    best_wins = None
    for key, val in node.items():
        if isinstance(val, dict):
            if best_move is None or val["plays"] > best_visits:
                best_move = key
                best_visits = val["plays"]
                best_wins = val["wins"]

    print("{} {}/{}".format(best_move, best_wins, best_visits))

def mcts(board):
    root = { "plays": 0, "wins": 0 }

    while True:
        select_move(board, root)
        statistics(root)

if __name__ == "__main__":
    board, _ = Board.from_epd("r4k2/p3nppp/3q4/2Np1b2/1r1P3P/5QP1/P4PB1/2R1R1K1 w - -")
    mcts(board)
