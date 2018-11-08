import time
import subprocess
from subprocess import PIPE
import re
from chess import Move

nodes = 0

TIME_LIMIT = 15


class Searcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.name = "Tensorflow"

    def pos_key(self, b):
        fen = b.fen()
        parts = fen.split(' ')
        return "{} {}".format(parts[0], parts[1])

    def eval_cached(self, b):
        key = self.pos_key(b)
        if (key in self.eval_cache):
            return self.eval_cache[key]
        else:
            eval = self.evaluator(b)
            self.eval_cache[key] = eval
            return eval


    def qsearch(self, b, alpha, beta, ply = 0):
        global nodes
        nodes += 1

        if b.is_insufficient_material():
            return 0

        # print("{} qsearch({}, {})".format("  " * ply, alpha, beta))
        score = self.eval_cached(b)

        if score >= beta:
            return score
        if score > alpha:
            alpha = score
        for move in b.generate_pseudo_legal_captures():
            if not b.is_legal(move):
                continue
            b.push(move)
            t = -self.qsearch(b, -beta, -alpha, ply + 1)
            b.pop()
            if (t > score):
                score = t
                if score >= beta:
                    return score
                if score > alpha:
                    alpha = score

        return score


    def move_score(self, move, board):
        board.push(move)
        score = self.eval_cached(board)
        board.pop()
        return score


    def next_move_old(self, board):
        key = self.pos_key(board)
        hash_move = None
        if key in self.move_cache:
            hash_move = self.move_cache[key]
            yield hash_move

        l = list(board.generate_legal_moves())
        l.sort(key = lambda m: self.move_score(m, board))
        for move in l:
            if move != hash_move:
                yield move

    def next_move(self, board):
        key = self.pos_key(board)
        hash_move = None
        if key in self.move_cache:
            hash_move = self.move_cache[key]
            yield hash_move

        l = list(board.generate_pseudo_legal_moves())
        for move in l:
            if move != hash_move:
                if board.is_legal(move):
                    yield move

    def search(self, b, alpha, beta, ply):
        if ply == 0:
            return self.qsearch(b, alpha, beta)

        global nodes
        nodes += 1

        if b.is_insufficient_material():
            return 0

        if b.is_check():
            ply += 1

        max_score = -1000
        best_move = None

        if ply > 2:
            b.push(Move.null())
            score = -self.search(b, -beta, -alpha, ply-2)
            b.pop()
            if score >= beta:
                return score

        for move in self.next_move(b):
            b.push(move)
            if b.is_fivefold_repetition():
                score = 0
            else:
                score = -self.search(b, -beta, -alpha, ply-1)
            b.pop()
            if score > max_score:
                max_score = score
                best_move = move
            if max_score >= beta:
                key = self.pos_key(b)
                self.move_cache[key] = move
                return max_score
            if max_score > alpha:
                alpha = max_score

        if best_move is None:
            if b.is_stalemate():
                return 0
            if b.is_checkmate():
                return -999
        else:
            key = self.pos_key(b)
            self.move_cache[key] = best_move

        return max_score


    def select_move(self, b):
        global nodes
        nodes = 0
        self.eval_cache = {}
        self.move_cache = {}

        l = list(b.generate_legal_moves())
        if len(l) == 1:
            return l[0]
        l.sort(key = lambda m: self.move_score(m, b))

        start_time = time.perf_counter()
        now = start_time

        for depth in range(1, 10):
            max_score = -1000
            best_move = None
            for move in l:
                b.push(move)
                if b.is_fivefold_repetition():
                    score = 0
                else:
                    score = -self.search(b, -1000, -max_score, depth-1)
                b.pop()
                now = time.perf_counter()

                if best_move is None or score > max_score:
                    max_score = score
                    best_move = move
                    l.remove(move)
                    l.insert(0, move)
                print("{}: [{}] {} with score {:.3f}, {} nodes/sec".format(
                    depth,
                    b.san(best_move),
                    b.san(move),
                    score,
                    int(nodes / (now - start_time))),
                    end = '\r')
                if (now - start_time) >= TIME_LIMIT:
                    break
            print("{}: {} in {:.1f} secs {:.3f}, {} nodes/sec                    ".format(
                depth,
                b.san(best_move),
                now - start_time,
                max_score,
                int(nodes / (now - start_time))))
            if (now - start_time) >= TIME_LIMIT:
                break

        print("==> {} with score {:.3f}, {} nodes/sec                ".format(
            b.san(best_move),
            max_score,
            int(nodes / (now - start_time))))
        return best_move


class AmySearcher:

    def __init__(self):
        self.name = "Amy"

    def select_move(self, b):
        p = subprocess.Popen('Amy', stdin=PIPE, stdout=PIPE)
        fen = b.fen()
        commands = "easy\nlevel fixed/2\nepd {}\nxboard\ngo\n".format(fen)
        out, err = p.communicate(bytes(commands, 'ASCII'))
        reply = out.decode('ASCII')
        m = re.search('move ([a-h][1-8][a-h][1-8][QRNB]?)', out.decode('ASCII'))
        return b.parse_uci(m[1].lower())
