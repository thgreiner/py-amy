import time
import subprocess
from subprocess import PIPE
import re

nodes = 0

TIME_LIMIT = 1

class Searcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def qsearch(self, b, alpha, beta, ply = 0):
        global nodes
        nodes += 1

        if b.is_checkmate():
            return -999
        if b.is_stalemate() or b.is_insufficient_material():
            return 0

        # print("{} qsearch({}, {})".format("  " * ply, alpha, beta))
        score = self.evaluator(b)

        if score >= beta or ply > 1:
            return score
        if score > alpha:
            alpha = score
        for move in b.generate_legal_captures():
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
        score = self.evaluator(board)
        board.pop()
        return score


    def search(self, b, alpha, beta, ply):
        if ply == 0:
            return self.qsearch(b, alpha, beta)

        global nodes
        nodes += 1

        if b.is_insufficient_material():
            return 0

        l = list(b.generate_legal_moves())
        if len(l) == 0:
            if b.is_stalemate():
                return 0
            if b.is_checkmate():
                return -999

        l.sort(key = lambda m: self.move_score(m, b))
        max_score = -1000
        for move in l:
            b.push(move)
            if b.is_fivefold_repetition():
                score = 0
            else:
                score = -self.search(b, -beta, -alpha, ply-1)
            b.pop()
            if score > max_score:
                max_score = score
            if max_score >= beta:
                return max_score
            if max_score > alpha:
                alpha = max_score
        return max_score


    def select_move(self, b):
        global nodes
        nodes = 0
        l = list(b.generate_legal_moves())
        if len(l) == 1:
            return l[0]
        l.sort(key = lambda m: self.move_score(m, b))
        it_start_time = time.perf_counter()
        for depth in range(1, 10):
            max = -1000
            best_move = None
            for move in l:
                b.push(move)
                if b.is_fivefold_repetition():
                    score = 0
                else:
                    score = -self.search(b, -1000, -max, depth-1)
                b.pop()
                end_time = time.perf_counter()

                if best_move is None or score > max:
                    max = score
                    best_move = move
                    l.remove(move)
                    l.insert(0, move)
                print("{}: [{}] {} with score {:.4f} nodes: {}, {} nodes/sec".format(
                    depth,
                    b.san(best_move), b.san(move), score, nodes, int(nodes / (end_time - it_start_time))),
                    end = '\r')
                it_end_time = time.perf_counter()
                if (it_end_time - it_start_time) >= TIME_LIMIT:
                    break
            print("{}: {} in {:.1f} secs                       ".format(
                depth, b.san(best_move), it_end_time - it_start_time))
            if (it_end_time - it_start_time) >= TIME_LIMIT:
                break

        print("==> {} with score {}                  ".format(b.san(best_move), max))
        return best_move


class AmySearcher:

    def select_move(self, b):
        p = subprocess.Popen('Amy', stdin=PIPE, stdout=PIPE)
        fen = b.fen()
        commands = "easy\nlevel fixed/2\nepd {}\nxboard\ngo\n".format(fen)
        out, err = p.communicate(bytes(commands, 'ASCII'))
        reply = out.decode('ASCII')
        m = re.search('move ([a-h][1-8][a-h][1-8][QRNB]?)', out.decode('ASCII'))
        return b.parse_uci(m[1].lower())
