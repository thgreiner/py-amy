import time
from chess import Move
from chess_input import BoardAndMoveRepr
import numpy as np
import random

TIME_LIMIT = 3


class TimeOutException(Exception):
    pass


class MctsSearcher:
    def __init__(self, move_model, score_model):
        self.move_model = move_model
        self.score_model = score_model
        self.name = "MCTS"
        self.repr = BoardAndMoveRepr()

    def pos_key(self, b):
        fen = b.fen()
        parts = fen.split(" ")
        return "{} {}".format(parts[0], parts[1])

    def select_index(self, moves, scores):
        sum_scores = sum(scores)
        r = random.uniform(0, sum_scores)
        tmp = 0
        for i in range(0, len(moves)):
            tmp += scores[i]
            if tmp >= r:
                break
        return i

    def next_move_mc(self, board, from_pred, to_pred):
        moves = list(board.generate_pseudo_legal_moves())
        while moves:
            scores = np.array(
                [self.move_score(from_pred, to_pred, board, m) for m in moves]
            )
            idx = self.select_index(moves, scores)
            if board.is_legal(moves[idx]):
                return moves[idx]
            moves.remove(moves[idx])
        return None

    def next_move(self, board, from_pred, to_pred):
        moves = list(board.generate_pseudo_legal_moves())
        if len(moves) == 0:
            return
        moves = sorted(
            moves, key=lambda m: -self.move_score(from_pred, to_pred, board, m)
        )

        while moves:
            move = moves[0]
            moves.remove(move)

            if board.is_legal(move):
                yield move
                break

        while moves:
            scores = np.array(
                [self.move_score(from_pred, to_pred, board, m) for m in moves]
            )
            idx = self.select_index(moves, scores)
            if board.is_legal(moves[idx]):
                yield moves[idx]
            moves.remove(moves[idx])

    def move_score(self, from_pred, to_pred, board, move):
        xor = 0
        if not board.turn:
            xor = 0x38
        type = board.piece_at(move.from_square).piece_type
        fr = move.from_square ^ xor
        to = move.to_square ^ xor
        return max(from_pred[fr], 0) * max(to_pred[type - 1][to], 0)

    def play_out(self, b):

        if b.is_game_over(True):
            print(b)
            return b.result(True)

        input = self.repr.board_to_array(b)
        predictions = self.move_model.predict([input.reshape(1, self.repr.SIZE)])
        from_pred = predictions[0].flatten()
        to_pred = [predictions[i].flatten() for i in range(1, 7)]

        move = self.next_move_mc(b, from_pred, to_pred)
        if move:
            # print(b)
            # print(b.san(move))
            # print()

            b.push(move)
            result = self.play_out(b)
            b.pop()
            return result
        else:
            return None

    def search(self, b, alpha, beta, depth):
        # print("search({}, {}, {})".format(alpha, beta, depth))
        self.nodes += 1

        if self.nodes > self.next_time_check:
            self.elapsed = time.perf_counter() - self.start_time
            if self.elapsed > self.time_limit:
                raise TimeOutException()
            self.next_time_check = self.nodes + 100

        incheck = False

        if len(b.move_stack) > self.limit:
            print("Limit reached.")
            return 0

        if b.is_insufficient_material() or b.is_fivefold_repetition():
            return 0

        if b.can_claim_draw() and 0 >= beta:
            return 0

        if b.is_check():
            depth += 1
            incheck = True

        input = self.repr.board_to_array(b)
        predictions = self.move_model.predict([input.reshape(1, self.repr.SIZE)])
        from_pred = predictions[0].flatten()
        to_pred = [predictions[i].flatten() for i in range(1, 7)]

        if depth <= 0:
            predictions = self.score_model.predict([input.reshape(1, self.repr.SIZE)])
            max_score = predictions[0].flatten()[0]
            if depth < -4:
                return max_score

            if max_score >= beta:
                return max_score
            if max_score > alpha:
                alpha = max_score
        else:
            max_score = -1000

        best_move = None

        max_moves = max(1, depth)
        i = 0
        for move in self.next_move(b, from_pred, to_pred):
            b.push(move)
            try:
                score = -self.search(b, -beta, -alpha, depth - 1)
            finally:
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
            i += 1
            if not incheck and i >= max_moves:
                break

        if best_move is None:
            if b.is_stalemate():
                return 0
            if b.is_checkmate():
                return -999
        else:
            key = self.pos_key(b)
            self.move_cache[key] = best_move

        return max_score

    def pv(self, b, move, depth=0):
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
        self.limit = len(b.move_stack) + 100

        l = list(b.generate_legal_moves())
        if len(l) == 1:
            return l[0]

        self.start_time = time.perf_counter()
        self.next_time_check = self.nodes + 100
        self.time_limit = TIME_LIMIT

        best_move = None
        try:
            for depth in range(1, 10):
                max_score = -1000
                for move in l:
                    b.push(move)
                    try:
                        if b.is_fivefold_repetition():
                            score = 0
                        else:
                            score = -self.search(b, -1000, -max_score, depth - 1)
                    finally:
                        b.pop()

                    self.elapsed = time.perf_counter() - self.start_time

                    print(
                        "{}: [{}] {} with score {:.3f}, {} nodes/sec".format(
                            depth,
                            b.san(best_move),
                            b.san(move),
                            score,
                            int(self.nodes / self.elapsed),
                        ),
                        end="\r",
                    )
                    if best_move is None or score > max_score:
                        max_score = score
                        best_move = move
                        l.remove(move)
                        l.insert(0, move)
                print(
                    "{}: {} in {:.1f} secs {:.3f}, {} nodes/sec                    ".format(
                        depth,
                        self.pv(b, best_move),
                        self.elapsed,
                        max_score,
                        int(self.nodes / self.elapsed),
                    )
                )
        except TimeOutException:
            pass

        print(
            "==> {} with score {:.3f}, {} nodes/sec                ".format(
                b.san(best_move), max_score, int(self.nodes / self.elapsed)
            )
        )
        return best_move
