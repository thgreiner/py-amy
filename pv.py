def pv(board, node, variation=None):

    if variation == None:
        variation = []

    if not node.expanded():
        return variation

    _, best_move = max(
        ((child.visit_count, action) for action, child in node.children.items()),
        key=lambda e: e[0],
    )

    variation.append(best_move)
    board.push(best_move)
    pv(board, node.children[best_move], variation)
    board.pop()
    return variation


def variations(board, move, child, count):

    vars = []
    prefix = []

    board.push(move)

    while True:
        stats = [
            (key, val) for key, val in child.children.items() if val.visit_count > 0
        ]

        if len(stats) != 1:
            break

        prefix.append(stats[0][0])
        child = stats[0][1]

    stats = sorted(stats, key=lambda e: e[1].visit_count, reverse=True)

    for m, grand_child in stats[:count]:
        line = []
        for mp in prefix:
            line.append(mp)
            board.push(mp)

        line.append(m)
        board.push(m)
        pv(board, grand_child, line)
        board.pop()

        for mp in prefix:
            board.pop()

        vars.append(board.variation_san(line))

    board.pop()
    return vars
