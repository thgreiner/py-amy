from chess.gaviota import open_tablebase
from chess import popcount
from prometheus_client import Counter
from threading import Lock

tb = open_tablebase("gtb")
tb_probe_counter = Counter("tb_probes", "Successful tablebase probes")

tb_lock = Lock()


def get_optimal_move(board):

    if (
        popcount(
            board.pawns | board.knights | board.bishops | board.rooks | board.queens
        )
        > 2
    ):
        return None, None

    best_move = None
    best_val = 0

    for m in board.generate_legal_moves():

        board.push(m)
        is_checkmate = board.is_checkmate()
        if not is_checkmate:
            try:
                with tb_lock:
                    val = -tb.probe_dtm(board)
            except KeyError:
                val = None
        board.pop()

        if is_checkmate:
            best_move = m
            break

        if val is None:
            continue

        tb_probe_counter.inc()

        if best_move is None:
            best_move = m
            best_val = val
        else:
            if best_val < 0:
                if val >= 0 or val < best_val:
                    best_move = m
                    best_val = val
            elif best_val > 0:
                if val > 0 and val < best_val:
                    best_move = m
                    best_val = val
            elif best_val == 0 and val > 0:
                best_move = m
                best_val = val

    return best_move, tb_val_to_str(best_val)


def tb_val_to_str(val):
    if val == 0:
        return "Draw"
    elif val > 0:
        return "M{}".format(val // 2)
    else:
        return "-M{}".format((-val + 1) // 2)
