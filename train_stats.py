from time import strftime

from prometheus_client import Gauge

loss_gauge = Gauge("training_loss", "Training loss")
moves_accuracy_gauge = Gauge("training_move_accuracy", "Move accuracy")
moves_top5_accuracy_gauge = Gauge("training_move_top5_accuracy", "Top 5 move accuracy")
score_mae_gauge = Gauge("training_score_mae", "Score mean absolute error")
wdl_accuracy_gauge = Gauge("wdl_accuracy", "WDL accuracy")


class Stats(object):
    def __init__(self):

        self.sum_moves_accuracy = 0
        self.sum_moves_top5_accuracy = 0
        self.sum_score_mae = 0
        self.sum_loss = 0
        self.sum_wdl_accuracy = 0
        self.sum_mlh = 0
        self.sum_cnt = 0

    def __call__(self, step_output, cnt):

        loss = step_output[0]
        moves_loss = step_output[1]
        score_loss = step_output[2]
        wdl_loss = step_output[3]
        mlh_loss = step_output[4]
        reg_loss = abs(loss - moves_loss - score_loss - 0.1 * (wdl_loss + mlh_loss))

        moves_accuracy = step_output[5]
        moves_top5_accuracy = step_output[6]
        score_mae = step_output[7]
        wdl_accuracy = step_output[8]
        mlh_mae = step_output[9]

        loss_gauge.set(loss)
        moves_accuracy_gauge.set(moves_accuracy * 100)
        moves_top5_accuracy_gauge.set(moves_top5_accuracy * 100)
        score_mae_gauge.set(score_mae)
        wdl_accuracy_gauge.set(wdl_accuracy * 100)

        self.sum_moves_accuracy += moves_accuracy * cnt
        self.sum_moves_top5_accuracy += moves_top5_accuracy * cnt
        self.sum_score_mae += score_mae * cnt
        self.sum_loss += loss * cnt
        self.sum_wdl_accuracy += wdl_accuracy * cnt
        self.sum_mlh += mlh_mae * cnt
        self.sum_cnt += cnt

        return "loss: {:.2f} = {:.2f} + {:.3f} + {:.3f} + {:.3f}, moves: {:4.1f}% top 5: {:4.1f}%, score: {:.2f}, wdl: {:4.1f}% || avg: {:.3f}, {:.2f}% top 5: {:.2f}%, {:.3f}, wdl: {:.2f}% mlh: {:.2f}".format(
            loss,
            moves_loss,
            score_loss,
            reg_loss,
            mlh_loss,
            moves_accuracy * 100,
            moves_top5_accuracy * 100,
            score_mae,
            wdl_accuracy * 100,
            self.sum_loss / self.sum_cnt,
            self.sum_moves_accuracy * 100 / self.sum_cnt,
            self.sum_moves_top5_accuracy * 100 / self.sum_cnt,
            self.sum_score_mae / self.sum_cnt,
            self.sum_wdl_accuracy * 100 / self.sum_cnt,
            self.sum_mlh / self.sum_cnt,
        )

    def write_to_file(self, model_name, filename="stats.txt"):

        with open(filename, "a") as statsfile:
            print(
                "{} [{}] {} positions: {:.3f}, {:.2f}% top 5: {:.2f}%, {:.3f}, wdl: {:.2f}%".format(
                    strftime("%Y-%m-%d %H:%M"),
                    model_name,
                    self.sum_cnt,
                    self.sum_loss / self.sum_cnt,
                    self.sum_moves_accuracy * 100 / self.sum_cnt,
                    self.sum_moves_top5_accuracy * 100 / self.sum_cnt,
                    self.sum_score_mae / self.sum_cnt,
                    self.sum_wdl_accuracy * 100 / self.sum_cnt,
                ),
                file=statsfile,
            )
