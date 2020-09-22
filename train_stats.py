class Stats(object):

    def __init__(self):

        self.sum_moves_accuracy = 0
        self.sum_score_mae = 0
        self.sum_loss = 0
        self.sum_cnt = 0

    def __call__(self, step_output, cnt):

        loss = step_output[0]
        moves_loss = step_output[1]
        score_loss = step_output[2]
        reg_loss = abs(loss - moves_loss - score_loss)

        moves_accuracy = step_output[3]
        score_mae = step_output[6]

        self.sum_moves_accuracy += moves_accuracy * cnt
        self.sum_score_mae += score_mae * cnt
        self.sum_loss += loss * cnt
        self.sum_cnt += cnt

        return "loss: {:.2f} = {:.2f} + {:.2f} + {:.2f}, move accuracy: {:4.1f}%, score mae: {:.2f} || avg: {:.3f}, {:.2f}%, {:.3f}".format(
            loss,
            moves_loss, score_loss, reg_loss,
            moves_accuracy * 100,
            score_mae,
            self.sum_loss / self.sum_cnt,
            self.sum_moves_accuracy * 100 / self.sum_cnt,
            self.sum_score_mae / self.sum_cnt
        )
 

    def write_to_file(self, filename="stats.txt"):

        with open(filename, "a") as statsfile:
            print("{} positions: {:.3f}, {:.2f}%, {:.3f}".format(
                    self.sum_cnt,
                    self.sum_loss / self.sum_cnt,
                    self.sum_moves_accuracy * 100 / self.sum_cnt,
                    self.sum_score_mae / self.sum_cnt),
                file=statsfile)
