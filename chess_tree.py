import chess.pgn
from chess import Board, Move, Piece

pgn = open("TWIC.pgn")

ngames = 0

class GameNode:
    def __init__(self):
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.moves = {}

    def update_stats(self, result):
        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        else:
            self.draws += 1

    def num_games(self):
        return self.white_wins + self.black_wins + self.draws

    def __str__(self):
        return "+{} ={} -{}".format(self.white_wins, self.draws, self.black_wins)

root = GameNode()

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

    for move in game.main_line():
        san = b.san(move)
        if san in d.moves:
            d = d.moves[san]
        else:
            next = GameNode()
            d.moves[san] = next
            d = next

        d.update_stats(result)
        b.push(move)

def traverse_game_tree(d, prefix=None, depth=0):
    if d.num_games() > 1 or len(d.moves) == 0:
        print("{} {}".format(prefix, d))
    for move in d.moves.keys():
        if prefix is None:
            traverse_game_tree(d.moves[move], move, depth+1)
        else:
            traverse_game_tree(d.moves[move], prefix + " " + move, depth+1)

traverse_game_tree(root)
