import chess
from chess import Board, Move, Piece
import random
import numpy as np
import chess.pgn
import time

# POSITIONS_TO_LEARN_APRIORI = 900000
POSITIONS_TO_LEARN_APRIORI = 562

SIZE_PER_COLOR = 49 + 5 * 65
SIZE = 2 * SIZE_PER_COLOR + 3

OPENING = -1

TIME_LIMIT = 1

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

model1 = keras.Sequential([
    keras.layers.Dense(200, input_shape=(SIZE, )),
#    keras.layers.Dropout(0.2),
    keras.layers.Dense(90),
#    keras.layers.Dropout(0.1),
    keras.layers.Dense(1)
])

model2 = keras.Sequential([
    keras.layers.Dense(512, input_shape=(SIZE, )),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(1)
])

opt1 = tf.train.AdamOptimizer()
opt2 = tf.train.RMSPropOptimizer(0.001)

model1.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])

model2.compile(optimizer=opt1,
               loss='mean_squared_error',
               metrics=['mae'])


def label_for_result(result):
    if result == '1-0':
        return 1
    if result == '0-1':
        return -1
    return 0


def phasing(label, moves_in_game, current_move):
    return label / (1.0 + min(30, moves_in_game - current_move))


def get_offset(square, piece_type):
    if piece_type == 1:
        return square - 8
    else:
        return 49 + (piece_type - 2) * 65 + square


eval_buf = np.ndarray((SIZE,))


def board_to_array(b):
    global eval_buf
    eval_buf[:] = 0

    total_pieces = 0
    for piece_type in range(1, 7):
        balance = 0
        squares = b.pieces(piece_type, True)
        for sq in squares:
            eval_buf[get_offset(sq, piece_type)] = 1
            balance += 1
            total_pieces += 1
        squares = b.pieces(piece_type, False)
        for sq in squares:
            eval_buf[get_offset(sq, piece_type) + SIZE_PER_COLOR] = 1
            balance -= 1
            total_pieces += 1
        eval_buf[get_offset(64, piece_type)] = balance
    if b.turn:
        eval_buf[SIZE - 2] = 1
    else:
        eval_buf[SIZE - 1] = 1
    eval_buf[SIZE - 3] = total_pieces
    return eval_buf


def evaluate(board, model):
    input = board_to_array(board)
    score = model.predict([input.reshape(1, SIZE)])[0][0]
    if not b.turn:
        score = -score
    score += random.uniform(-0.005, 0.005)
    return score


nodes = 0


def qsearch(b, alpha, beta, model, ply = 0):
    global nodes
    nodes += 1

    if b.is_checkmate():
        return -999
    if b.is_stalemate() or b.is_insufficient_material():
        return 0

    # print("{} qsearch({}, {})".format("  " * ply, alpha, beta))
    score = evaluate(b, model)

    if score >= beta or ply > 1:
        return score
    if score > alpha:
        alpha = score
    for move in b.generate_legal_captures():
        b.push(move)
        t = -qsearch(b, -beta, -alpha, model, ply + 1)
        b.pop()
        if (t > score):
            score = t
            if score >= beta:
                return score
            if score > alpha:
                alpha = score

    return score


def move_score(move, board, model):
    board.push(move)
    score = evaluate(board, model)
    board.pop()
    return score


def search(b, alpha, beta, model, ply):
    if ply == 0:
        return qsearch(b, alpha, beta, model)

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

    l.sort(key = lambda m: move_score(m, b, model))
    max_score = -1000
    for move in l:
        b.push(move)
        if b.is_fivefold_repetition():
            score = 0
        else:
            score = -search(b, -beta, -alpha, model, ply-1)
        b.pop()
        if score > max_score:
            max_score = score
        if max_score >= beta:
            return max_score
        if max_score > alpha:
            alpha = max_score
    return max_score


def select_move(b, model):
    global nodes
    l = list(b.generate_legal_moves())
    if len(l) == 1:
        return l[0]
    l.sort(key = lambda m: move_score(m, b, model))
    it_start_time = time.perf_counter()
    for depth in range(1, 10):
        max = -1000
        best_move = None
        for move in l:
            nodes = 0
            start_time = time.perf_counter()
            b.push(move)
            if b.is_fivefold_repetition():
                score = 0
            else:
                score = -search(b, -1000, -max, model, depth-1)
            b.pop()
            end_time = time.perf_counter()

            if best_move is None or score > max:
                max = score
                best_move = move
                l.remove(move)
                l.insert(0, move)
            print("{}: [{}] {} with score {:.4f} nodes: {}, {} nodes/sec".format(
                depth,
                b.san(best_move), b.san(move), score, nodes, int(nodes / (end_time - start_time))),
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

def gen_kqk():
    while True:
        pos = random.sample(list(range(64)), 3)
        b = Board()
        b.clear()
        b.set_piece_at(pos[0], Piece.from_symbol('K'))
        b.set_piece_at(pos[1], Piece.from_symbol(random.choice(['Q', 'q'])))
        b.set_piece_at(pos[2], Piece.from_symbol('k'))
        
        if b.status() == chess.STATUS_VALID:
            return b
        
pgn = open("mate.pgn")

npos = POSITIONS_TO_LEARN_APRIORI
train_data = np.zeros((npos, SIZE))
train_labels = np.zeros((npos))

i = 0
while True:
    game = chess.pgn.read_game(pgn)
    label = label_for_result(game.headers["Result"])
    b = game.board()
    nmoves = 0
    moves_in_game = len(list(game.main_line()))
    for move in game.main_line():
        is_capture = b.is_capture(move)
        if nmoves > OPENING and not is_capture:
            train_data[i] = board_to_array(b)
            train_labels[i] = phasing(label, moves_in_game, nmoves)
            i += 1
            train_data[i] = board_to_array(b.mirror())
            train_labels[i] = -phasing(label, moves_in_game, nmoves)
            i += 1
        b.push(move)
        nmoves += 1
        if i >= npos:
            break
    if i >= npos:
        break
    print(i, end='\n')

print(i)

offset = npos
while True:

    model1.fit(train_data, train_labels, batch_size=128, epochs=30)
    model2.fit(train_data, train_labels, batch_size=128, epochs=30)

    # b = Board()
    b = gen_kqk()
    while not b.is_game_over() and len(b.move_stack) < 30:
        if len(b.move_stack) > OPENING:
            if b.turn:
                move = select_move(b, model2)
            else:
                move = select_move(b, model2)
        else:
            move = random.choice(list(b.generate_legal_moves()))

        print(b.san(move))
        b.push(move)
        print(b)
        print(b.fen())

    print("{} after {} moves.".format(b.result(), b.fullmove_number))

    result = label_for_result(b.result())
    print(result)

    n = 2 * (len(b.move_stack) - OPENING)

    if offset == 0:
        train_data = np.zeros((n, SIZE))
        train_labels = np.zeros((n))
    else:
        new_train_data = np.zeros((n + offset, SIZE))
        new_train_labels = np.zeros((n + offset))
        new_train_data[0:offset] = train_data
        new_train_labels[0:offset] = train_labels
        train_data = new_train_data
        train_labels = new_train_labels

    moves_in_game = len(b.move_stack)
    while len(b.move_stack) > OPENING:
        try:
            m = b.pop()
            i = 2 * (len(b.move_stack) - OPENING)
            train_data[offset + i] = board_to_array(b)
            train_labels[offset + i] = phasing(result, moves_in_game, len(b.move_stack))
            i += 1
            train_data[offset + i] = board_to_array(b.mirror())
            train_labels[offset + i] = -phasing(result, moves_in_game, len(b.move_stack))
        except:
            break

    offset += n
