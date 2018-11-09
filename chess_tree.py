import chess.pgn
from chess import Board, Move, Piece
from chess_input import Repr1, Repr2
import numpy as np

def phasing(label, moves_in_game, current_move):
    return 10 * label * (1.0 + moves_in_game - current_move) ** -0.8

repr = Repr1()

pgn = open("input.pgn")

ngames = 0

class GameNode:
    def __init__(self):
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.longest_game = 0
        self.moves = {}

    def update_stats(self, result):
        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        else:
            self.draws += 1
    
    def update_longest_game(self, nmoves):
        self.longest_game = max(self.longest_game, nmoves)

    def num_games(self):
        return self.white_wins + self.black_wins + self.draws

    def __str__(self):
        return "+{} ={} -{} ({} moves)".format(
            self.white_wins, self.draws, self.black_wins,
            self.longest_game)

root = GameNode()
nodes = 1

while True:

    try:
        game = chess.pgn.read_game(pgn)
    except UnicodeDecodeError or ValueError:
        continue

    if game is None:
        break

    result = game.headers["Result"]
    ngames += 1
    if ngames % 1000 == 0:
        print(ngames, flush=True)
    b = game.board()
    d = root

    nmoves = len(list(game.main_line()))

    for move in game.main_line():
        san = b.san(move)
        if san in d.moves:
            d = d.moves[san]
        else:
            next = GameNode()
            d.moves[san] = next
            d = next
            nodes += 1

        d.update_stats(result)
        d.update_longest_game(nmoves)

        b.push(move)

train_data = np.zeros((nodes, repr.SIZE))
train_labels = np.zeros((nodes))
i = 0

def traverse_game_tree(d, depth=0):
    global i
    for move in d.moves.keys():
        m = board.parse_san(move)
        is_capture = board.is_capture(m)
        board.push(m)
        next_node = d.moves[move]
        
        if not is_capture and depth > 8:
            train_data[i] = repr.board_to_array(board)
            if board.turn:
                label = (next_node.white_wins - next_node.black_wins) / next_node.num_games()
            else:
                label = (next_node.black_wins - next_node.white_wins) / next_node.num_games()
            train_labels[i] = phasing(label, next_node.longest_game, depth)
            i += 1
            
        traverse_game_tree(next_node, depth+1)
        board.pop()

board = Board()
traverse_game_tree(root)

np.savez('input.npz', data = train_data[:i], labels= train_labels[:i])
