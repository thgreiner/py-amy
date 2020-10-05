import time
import subprocess
from subprocess import PIPE
import re
from chess import Move
import chess
import numpy as np

nodes = 0

TIME_LIMIT = 1000
NODE_LIMIT = 100000

EPSILON = 0.01

class TimeOutException(Exception):
    pass

class Searcher:
    def __init__(self, evaluator, name="Tensorflow"):
        self.evaluator = evaluator
        self.name = name


    def pos_key(self, b):
        fen = b.fen()
        parts = fen.split(' ')
        return "{} {}".format(parts[0], parts[1])



    def history_index(self, m):
        return (m.from_square << 6) + m.to_square


    def eval_cached(self, b):
        key = self.pos_key(b)
        if (key in self.eval_cache):
            return self.eval_cache[key]
        else:
            eval = self.evaluator(b)
            self.eval_cache[key] = eval
            return eval


    def next_capture(self, board):
        victims = [
            board.pieces_mask(chess.QUEEN, not board.turn),
            board.pieces_mask(chess.ROOK, not board.turn),
            board.pieces_mask(chess.KNIGHT, not board.turn) | board.pieces_mask(chess.BISHOP, not board.turn),
            board.pieces_mask(chess.PAWN, not board.turn),
        ]
        attackers = [
            board.pieces_mask(chess.PAWN, board.turn),
            board.pieces_mask(chess.KNIGHT, board.turn) | board.pieces_mask(chess.BISHOP, board.turn),
            board.pieces_mask(chess.ROOK, board.turn),
            board.pieces_mask(chess.QUEEN, board.turn),
            board.pieces_mask(chess.KING, board.turn)
        ]
        for victim in range(0, 4):
            victims_mask = victims[victim]
            for attacker in range(0, 4-victim):
                attackers_mask = attackers[attacker]
                captures = board.generate_pseudo_legal_captures(attackers_mask, victims_mask)
                for move in captures:
                    if board.is_legal(move):
                        yield move
        for victim in range(0, 4):
            victims_mask = victims[victim]
            for attacker in range(4-victim, len(attackers)):
                attackers_mask = attackers[attacker]
                captures = board.generate_pseudo_legal_captures(attackers_mask, victims_mask)
                for move in captures:
                    if board.is_legal(move):
                        yield move


    def qsearch(self, b, alpha, beta, ply = 0):
        self.nodes += 1

        if b.is_insufficient_material():
            return 0

        # print("{} qsearch({}, {})".format("  " * ply, alpha, beta))
        score = self.eval_cached(b)

        if score >= beta:
            return score
        if score > alpha:
            alpha = score
        for move in self.next_capture(b):
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

        captures_searched = set()
        victims = [
            board.pieces_mask(chess.QUEEN, not board.turn),
            board.pieces_mask(chess.ROOK, not board.turn),
            board.pieces_mask(chess.KNIGHT, not board.turn) | board.pieces_mask(chess.BISHOP, not board.turn),
            board.pieces_mask(chess.PAWN, not board.turn),
        ]
        attackers = [
            board.pieces_mask(chess.PAWN, board.turn),
            board.pieces_mask(chess.KNIGHT, board.turn) | board.pieces_mask(chess.BISHOP, board.turn),
            board.pieces_mask(chess.ROOK, board.turn),
            board.pieces_mask(chess.QUEEN, board.turn),
            board.pieces_mask(chess.KING, board.turn)
        ]
        for victim in range(0, 4):
            victims_mask = victims[victim]
            for attacker in range(0, 4-victim):
                attackers_mask = attackers[attacker]
                captures = board.generate_pseudo_legal_captures(attackers_mask, victims_mask)
                for move in captures:
                    if move != hash_move and board.is_legal(move):
                        captures_searched.add(move)
                        yield move

        # for victim in range(0, 4):
        #     victims_mask = victims[victim]
        #     for attacker in range(4-victim, len(attackers)):
        #         attackers_mask = attackers[attacker]
        #         captures = board.generate_pseudo_legal_captures(attackers_mask, victims_mask)
        #         for move in captures:
        #             if move != hash_move and board.is_legal(move):
        #                 captures_searched.add(move)
        #                 yield move

        l = list(board.generate_pseudo_legal_moves())
        l = sorted(l, key = lambda m : self.history_table[self.history_index(m)], reverse = True)

        for move in l:
            if move != hash_move and not move in captures_searched:
                if board.is_legal(move):
                    yield move

    def search(self, b, alpha, beta, depth):
        if depth == 0:
            return self.qsearch(b, alpha, beta)

        self.nodes += 1

        if self.nodes > self.next_time_check:
            self.elapsed = time.perf_counter() - self.start_time
            if self.elapsed > self.time_limit or self.nodes > NODE_LIMIT:
                raise TimeOutException()
            self.next_time_check = self.nodes + 100

        if b.is_insufficient_material():
            return 0

        if b.can_claim_draw() and 0 >= beta:
            return 0

        if b.is_check():
            depth += 1

        max_score = -1000
        best_move = None

        #if ply > 2:
        #    b.push(Move.null())
        #    try:
        #        score = -self.search(b, -beta, -alpha, ply-2)
        #    finally:
        #        b.pop()
        #    if score >= beta:
        #        return score

        for move in self.next_move(b):
            b.push(move)
            try:
                if b.is_fivefold_repetition():
                    score = 0
                else:
                    score = -self.search(b, -beta, -alpha, depth-1)
            finally:
                b.pop()

            if score > max_score:
                max_score = score
                best_move = move
            if max_score >= beta:
                key = self.pos_key(b)
                self.move_cache[key] = move
                self.history_table[self.history_index(move)] += (1 << depth)
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


    def pv(self, b, move, depth = 0):
        line = b.san(move)
        if depth < 10:
            b.push(move)
            key = self.pos_key(b)
            hash_move = None
            if key in self.move_cache:
                hash_move = self.move_cache[key]
                line += " " + self.pv(b, hash_move, depth + 1)
            b.pop()
        return line

    def select_move(self, b):
        self.nodes = 0

        self.eval_cache = {}
        self.move_cache = {}
        self.history_table = np.zeros((4096), np.int32)

        l = list(b.generate_legal_moves())
        if len(l) == 1:
            return l[0]
        l.sort(key = lambda m: self.move_score(m, b))

        self.start_time = time.perf_counter()
        self.next_time_check = self.nodes + 100
        self.time_limit = TIME_LIMIT
        self.elapsed = 0

        best_move = None
        max_score = 0

        try:
            for depth in range(1, 10):
                is_pv = True
                for move in l:
                    san = b.san(move)
                    print("{:2d}  {:5.1f}          {}     ".format(
                        depth,
                        self.elapsed,
                        san), end = '\r')
                        
                    b.push(move)
                    try:
                        alpha = max_score
                        if is_pv:
                            alpha = max_score - EPSILON
                        beta = max_score + EPSILON
                        if b.is_fivefold_repetition():
                            score = 0
                        else:
                            score = -self.search(b, -beta, -alpha, depth-1)
                            if is_pv and score <= alpha:
                                if depth > 1:
                                    print("{:2d}- {:5.1f}  {:+.3f}  {}".format(
                                        depth,
                                        self.elapsed,
                                        score,
                                        san))
                                score = -self.search(b, -score, 1000, depth-1)
                            if score >= beta:
                                best_move = move
                                if depth > 1:
                                    print("{:2d}+ {:5.1f}  {:+.3f}  {}".format(
                                        depth,
                                        self.elapsed,
                                        score,
                                        san))
                                score = -self.search(b, -1000, -score, depth-1)
                                
                    finally:
                        b.pop()

                    self.elapsed = time.perf_counter() - self.start_time

                    if is_pv or score > max_score:
                        max_score = score
                        best_move = move
                        l.remove(move)
                        l.insert(0, move)
                            
                        if depth > 1:
                            print("{:2d}+ {:5.1f}  {:+.3f}  {}".format(
                                depth,
                                self.elapsed,
                                score,
                                self.pv(b, best_move)))
                    is_pv = False

                print("{:2d}  {:5.1f}  {:+.3f}  {}, {} nodes/sec                    ".format(
                    depth,
                    self.elapsed,
                    max_score,
                    self.pv(b, best_move),
                    int(self.nodes / self.elapsed)))
        except TimeOutException:
            pass

        print("==> {} with score {:.3f}, {} nodes/sec                ".format(
            b.san(best_move),
            max_score,
            int(self.nodes / self.elapsed)))
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
