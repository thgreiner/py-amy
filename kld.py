from math import log


class KLD:
    def __init__(self):
        self.last_p = None

    def update(self, tree):

        p = {key: value.visit_count for key, value in tree.children.items()}
        visit_sum = sum(p.values())
        p = {key: value / visit_sum for key, value in p.items()}

        kld = None

        if self.last_p is not None:
            kld = 0.0
            for key in self.last_p.keys():
                if self.last_p[key] != 0.0:
                    kld += self.last_p[key] * log(self.last_p[key] / p[key])

        self.last_p = p
        return kld
