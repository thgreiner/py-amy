from math import sqrt, log

FORCED_PLAYOUT = 10000


class UCB:
    def __init__(self, pb_c_init, pb_c_base=19652):
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base

    def __call__(self, parent, child, forced_playouts=False):
        if forced_playouts:
            n_forced_playouts = sqrt(child.prior * parent.visit_count * 2)
            if child.visit_count < n_forced_playouts:
                return FORCED_PLAYOUT

        pb_c = log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base)
        pb_c += self.pb_c_init
        pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()

        return prior_score + value_score

    def __str__(self):
        return f"ucb(pb_c_init={self.pb_c_init})"
