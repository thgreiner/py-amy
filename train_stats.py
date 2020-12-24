from time import strftime


class Stats(object):
    def __init__(self):

        self.sum_moves_accuracy = 0
        self.sum_moves_top5_accuracy = 0
        self.sum_score_mae = 0
        self.sum_loss = 0
        self.sum_cnt = 0

    def __call__(self, step_output, cnt):

        loss = step_output[0]
        moves_loss = step_output[1]
        score_loss = step_output[2]
        reg_loss = abs(loss - moves_loss - score_loss)

        moves_accuracy = step_output[3]
        moves_top5_accuracy = step_output[4]
        score_mae = step_output[5]

        self.sum_moves_accuracy += moves_accuracy * cnt
        self.sum_moves_top5_accuracy += moves_top5_accuracy * cnt
        self.sum_score_mae += score_mae * cnt
        self.sum_loss += loss * cnt
        self.sum_cnt += cnt

        return "loss: {:.2f} = {:.2f} + {:.3f} + {:.3f}, moves: {:4.1f}% top 5: {:4.1f}%, score: {:.2f} || avg: {:.3f}, {:.2f}% top 5: {:.2f}%, {:.3f}".format(
            loss,
            moves_loss,
            score_loss,
            reg_loss,
            moves_accuracy * 100,
            moves_top5_accuracy * 100,
            score_mae,
            self.sum_loss / self.sum_cnt,
            self.sum_moves_accuracy * 100 / self.sum_cnt,
            self.sum_moves_top5_accuracy * 100 / self.sum_cnt,
            self.sum_score_mae / self.sum_cnt,
        )

    def write_to_file(self, model_name, filename="stats.txt"):

        with open(filename, "a") as statsfile:
            print(
                "{} [{}] {} positions: {:.3f}, {:.2f}% top 5: {:.2f}%, {:.3f}".format(
                    strftime("%Y-%m-%d %H:%M"),
                    model_name,
                    self.sum_cnt,
                    self.sum_loss / self.sum_cnt,
                    self.sum_moves_accuracy * 100 / self.sum_cnt,
                    self.sum_moves_top5_accuracy * 100 / self.sum_cnt,
                    self.sum_score_mae / self.sum_cnt,
                ),
                file=statsfile,
            )
